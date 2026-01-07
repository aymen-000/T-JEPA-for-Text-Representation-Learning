"""
Text-JEPA Fine-tuning Evaluation (CLS-based)

Evaluates a pretrained Text-JEPA encoder by FINE-TUNING
the entire encoder (not frozen) on AG News.
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
# Fine-tuning Model (CLS-based)
# -------------------------------------------------------
class TextFineTuneModel(nn.Module):
    """
    Fine-tuning model on top of a Text-JEPA encoder:
    [CLS] → LayerNorm → Dropout → Linear
    Encoder is NOT frozen - all parameters are trainable.
    """
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Encoder is NOT frozen - all parameters trainable

    def forward(self, input_ids):
        feats = self.encoder(input_ids)  # [B, L, D]
        cls_feat = feats[:, 0]  # CLS token

        cls_feat = self.norm(cls_feat)
        cls_feat = self.dropout(cls_feat)

        return self.classifier(cls_feat)


# -------------------------------------------------------
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    def __init__(self, output_dir="outputs/text_jepa"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            output_dir, f"finetune_results_{timestamp}.csv"
        )

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "val_acc", "is_best"]
            )

        print(f"✓ CSV log created at: {self.csv_path}")

    def log(self, epoch, train_loss, train_acc, val_acc, is_best):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{train_acc:.2f}",
                    f"{val_acc:.2f}",
                    int(is_best),
                ]
            )


# -------------------------------------------------------
# Helper: detect tokenizer from checkpoint
# -------------------------------------------------------
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
# Fine-tuning Evaluation
# -------------------------------------------------------
def finetune_encoder(
    encoder_path,
    config_path,
    model_name=None,
    num_classes=4,
    batch_size=64,
    num_epochs=20,
    lr=2e-5,
    encoder_lr=1e-5,
    device="cuda",
    output_dir="outputs/text_jepa",
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger = CSVLogger(output_dir)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    checkpoint = torch.load(encoder_path, map_location=device)

    vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    embed_dim = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[1]
    max_seq_len = checkpoint["encoder"]["pos_embed"].shape[1]

    # Infer depth
    depth = max(
        int(k.split(".")[1]) + 1
        for k in checkpoint["encoder"]
        if k.startswith("blocks.") and ".norm1.weight" in k
    )

    num_heads = 8  # safe default

    # Tokenizer
    if model_name is None:
        model_name = get_model_name_from_checkpoint(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    assert tokenizer.cls_token_id is not None, \
        "Tokenizer must have a CLS token"

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
    encoder.train()  # Set to training mode

    # Fine-tuning model
    model = TextFineTuneModel(
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=num_classes,
    ).to(device)

    # Dataset
    dataset = load_dataset("ag_news")

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config["mask"]["max_tokens"],
        )
        out["labels"] = batch["label"]
        return out

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "labels"],
    )

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset["test"], batch_size=batch_size
    )

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.norm.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    print(f"Fine-tuning with encoder_lr={encoder_lr}, classifier_lr={lr}")

    # Training
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
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

        logger.log(
            epoch + 1,
            loss_sum / len(train_loader),
            train_acc,
            val_acc,
            is_best,
        )

        print(
            f"Epoch {epoch+1}: "
            f"Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%"
        )

    print(f"\nBEST FINE-TUNING ACCURACY: {best_acc:.2f}%")
    return best_acc


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Text-JEPA Fine-tuning (CLS)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/text_jepa")

    args = parser.parse_args()

    finetune_encoder(
        encoder_path=args.checkpoint,
        config_path=args.config,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        device=args.device,
        output_dir=args.output_dir,
    )
