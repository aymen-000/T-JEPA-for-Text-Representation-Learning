"""
Text-JEPA Linear Probing on Sentiment Analysis (SST-2)

Binary sentiment classification task from GLUE benchmark.
Encoder is FROZEN - only the MLP head is trained.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import csv
import os
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Linear Probe Model
# -------------------------------------------------------
class TextLinearProbeModel(nn.Module):
    """
    Linear probing model on top of a frozen Text-JEPA encoder:
    Mean pool over non-padding tokens → LayerNorm → Dropout → Linear
    Encoder is FROZEN - only the head is trainable.
    """
    def __init__(self, encoder, embed_dim, num_classes, pad_id=0):
        super().__init__()
        self.encoder = encoder
        self.pad_id = pad_id  # FIX: was used in forward() but never defined
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        with torch.no_grad():
            feats = self.encoder(input_ids)   # [B, L, D]

        # Mean pool over non-padding tokens
        mask = (input_ids != self.pad_id).unsqueeze(-1)  # [B, L, 1]

        feats = feats * mask
        sent_feat = feats.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, D]

        sent_feat = self.norm(sent_feat)
        sent_feat = self.dropout(sent_feat)
        return self.classifier(sent_feat)


# -------------------------------------------------------
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    def __init__(self, output_dir="outputs/sentiment"):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            output_dir, f"sentiment_linearprobe_results_{self.timestamp}.csv"
        )
        self.txt_path = os.path.join(
            output_dir, f"sentiment_linearprobe_log_{self.timestamp}.txt"
        )

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "val_acc", "is_best"]
            )

        print(f"✓ CSV log created at: {self.csv_path}")
        print(f"✓ Text log created at: {self.txt_path}")

    def log(self, epoch, train_loss, train_acc, val_acc, is_best):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                 f"{val_acc:.2f}", int(is_best)]
            )


def get_model_name_from_checkpoint(checkpoint):
    vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    if vocab_size == 30522:
        return "bert-base-uncased"
    elif vocab_size == 50257:
        return "gpt2"
    elif vocab_size == 32000:
        return "t5-base"
    return None


# -------------------------------------------------------
# Linear Probing on SST-2
# -------------------------------------------------------
def linearprobe_sentiment(
    encoder_path,
    config_path,
    model_name=None,
    batch_size=32,
    num_epochs=10,
    lr=0.01,
    device="cuda",
    output_dir="outputs/sentiment",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = datetime.now()
    logger = CSVLogger(output_dir)

    # Load config and checkpoint
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint = torch.load(encoder_path, map_location=device)

    vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    embed_dim = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[1]
    max_seq_len = checkpoint["encoder"]["pos_embed"].shape[1]

    depth = max(
        int(k.split(".")[1]) + 1
        for k in checkpoint["encoder"]
        if k.startswith("blocks.") and ".norm1.weight" in k
    )
    num_heads = 8

    if model_name is None:
        model_name = get_model_name_from_checkpoint(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Init encoder
    encoder, _ = init_model(
        device=device,
        model_name=model_name,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        pred_depth=config["meta"]["pred_depth"],
        pred_emb_dim=config["meta"]["pred_emb_dim"],
    )

    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()  # Frozen encoder stays in eval mode always

    # Linear probe model (binary classification)
    model = TextLinearProbeModel(
        encoder=encoder,
        embed_dim=embed_dim,
        pad_id=tokenizer.pad_token_id,
        num_classes=2,  # Negative / Positive
    ).to(device)

    # Count trainable vs frozen params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable:,}  |  Frozen parameters: {frozen:,}")

    # Load SST-2 dataset
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    def tokenize(batch):
        out = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=config["mask"]["max_tokens"],
        )
        out["labels"] = batch["label"]
        return out

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)

    # Optimizer: head only
    optimizer = torch.optim.AdamW([
        {'params': model.norm.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=lr, weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    print(f"Linear probing SST-2 with head lr={lr} (encoder is frozen)")

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.encoder.eval()       # Always keep encoder in eval
        model.norm.train()
        model.classifier.train()

        correct = total = 0
        loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        logger.log(epoch + 1, loss_sum / len(train_loader), train_acc, val_acc, is_best)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

    print(f"\nBEST SENTIMENT LINEAR PROBE ACCURACY: {best_acc:.2f}%")

    # Save model
    end_time = datetime.now()
    total_time = end_time - start_time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f"sentiment_linearprobe_{timestamp}.pth")

    torch.save({
        'head_state_dict': {
            'norm': model.norm.state_dict(),
            'classifier': model.classifier.state_dict(),
        },
        'best_accuracy': best_acc,
        'num_classes': 2,
        'embed_dim': embed_dim,
        'config': config,
    }, model_save_path)

    print(f"✓ Linear probe head saved to: {model_save_path}")

    # Save training log
    with open(logger.txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SENTIMENT ANALYSIS (SST-2) LINEAR PROBING LOG\n")
        f.write("=" * 70 + "\n\n")

        f.write("EXPERIMENT INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Task:                          Binary Sentiment Classification (SST-2)\n")
        f.write(f"Dataset:                       Stanford Sentiment Treebank v2\n")
        f.write(f"Start Time:                    {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time:                      {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Training Time:           {total_time}\n")
        f.write(f"Total Training Time (seconds): {total_time.total_seconds():.2f}s\n")
        f.write(f"Total Training Time (minutes): {total_time.total_seconds()/60:.2f}m\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Encoder Path:       {encoder_path}\n")
        f.write(f"Model Name:         {model_name}\n")
        f.write(f"Embedding Dim:      {embed_dim}\n")
        f.write(f"Max Sequence Len:   {max_seq_len}\n")
        f.write(f"Depth (layers):     {depth}\n")
        f.write(f"Number of Heads:    {num_heads}\n")
        f.write(f"Number of Classes:  2 (Negative / Positive)\n")
        f.write(f"Trainable Params:   {trainable:,}\n")
        f.write(f"Frozen Params:      {frozen:,}\n\n")

        f.write("TRAINING HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Batch Size:         {batch_size}\n")
        f.write(f"Number of Epochs:   {num_epochs}\n")
        f.write(f"Head Learning Rate: {lr}\n")
        f.write(f"Weight Decay:       0.01\n")
        f.write(f"Dropout:            0.2\n")
        f.write(f"Optimizer:          AdamW (head only)\n")
        f.write(f"Loss Function:      CrossEntropyLoss\n")
        f.write(f"Device:             {device}\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train Samples:      {len(dataset['train'])}\n")
        f.write(f"Validation Samples: {len(dataset['validation'])}\n")
        f.write(f"Train Batches:      {len(train_loader)}\n")
        f.write(f"Val Batches:        {len(val_loader)}\n\n")

        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n\n")

        f.write("SAVED FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Head Checkpoint:    {model_save_path}\n")
        f.write(f"CSV Results:        {logger.csv_path}\n")
        f.write(f"Training Log:       {logger.txt_path}\n\n")

        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write("TextLinearProbeModel(\n")
        f.write("  Encoder (Text-JEPA pretrained, FROZEN)\n")
        f.write("  LayerNorm                  ← trainable\n")
        f.write("  Dropout(0.2)               ← trainable\n")
        f.write(f"  Linear({embed_dim} -> 2)       ← trainable\n")
        f.write(")\n\n")

        f.write("NOTES\n")
        f.write("-" * 70 + "\n")
        f.write("- Encoder is FROZEN throughout training\n")
        f.write("- Only LayerNorm + Linear head are trained\n")
        f.write("- Encoder kept in eval() mode to disable dropout/batchnorm updates\n")
        f.write("- torch.no_grad() wraps encoder forward pass for efficiency\n")
        f.write("- CLS token used for classification\n")
        f.write("- encoder_lr argument removed (encoder not updated)\n\n")

        f.write("=" * 70 + "\n")
        f.write("END OF TRAINING LOG\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Training log saved to: {logger.txt_path}")
    print(f"✓ Total training time: {total_time}")

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sentiment Analysis Linear Probing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/sentiment")

    args = parser.parse_args()

    linearprobe_sentiment(
        encoder_path=args.checkpoint,
        config_path=args.config,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )