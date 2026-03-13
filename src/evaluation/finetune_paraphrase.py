"""
Text-JEPA Fine-tuning on Paraphrase Detection

Binary classification: determine if two sentences are paraphrases.
Can use either QQP (Quora Question Pairs) or MRPC (Microsoft Research Paraphrase Corpus)

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
from sklearn.metrics import f1_score, precision_score, recall_score

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Fine-tuning Model for Sentence-Pair Tasks
# -------------------------------------------------------
class SentencePairModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes, pad_id=0):
        super().__init__()
        self.encoder = encoder
        self.pad_id = pad_id
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        # Concatenate + element-wise difference + element-wise product
        self.classifier = nn.Linear(embed_dim * 4, num_classes)

        # Freeze the encoder — only MLP head is trained
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, sent1_ids, sent2_ids):
        # Encode both sentences (no grad since encoder is frozen)
        with torch.no_grad():
            sent1_feats = self.encoder(sent1_ids)   # [B, L, D]
            sent2_feats = self.encoder(sent2_ids)   # [B, L, D]

        # Get CLS tokens
        sent1_cls = sent1_feats[:, 0]
        sent2_cls = sent2_feats[:, 0]

        # Normalize
        sent1_cls = self.norm(sent1_cls)
        sent2_cls = self.norm(sent2_cls)

        # Rich interaction features
        diff = torch.abs(sent1_cls - sent2_cls)
        prod = sent1_cls * sent2_cls

        # Concatenate all features
        combined = torch.cat([sent1_cls, sent2_cls, diff, prod], dim=-1)
        combined = self.dropout(combined)

        return self.classifier(combined)


# -------------------------------------------------------
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    def __init__(self, output_dir="outputs/paraphrase"):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            output_dir, f"paraphrase_results_{self.timestamp}.csv"
        )
        self.txt_path = os.path.join(
            output_dir, f"paraphrase_log_{self.timestamp}.txt"
        )

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "val_acc",
                 "val_f1", "val_precision", "val_recall", "is_best"]
            )

        print(f"✓ CSV log created at: {self.csv_path}")
        print(f"✓ Text log created at: {self.txt_path}")

    def log(self, epoch, train_loss, train_acc, val_acc, val_f1,
            val_precision, val_recall, is_best):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                 f"{val_acc:.2f}", f"{val_f1:.4f}",
                 f"{val_precision:.4f}", f"{val_recall:.4f}", int(is_best)]
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
# Linear Probing on Paraphrase Detection
# -------------------------------------------------------
def finetune_paraphrase(
    encoder_path,
    config_path,
    dataset_name="mrpc",  # or "qqp"
    model_name=None,
    batch_size=32,
    num_epochs=15,
    lr=2e-5,
    device="cuda",
    output_dir="outputs/paraphrase",
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

    # Model with frozen encoder
    model = SentencePairModel(
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=2,  # Binary: paraphrase or not
        pad_id=tokenizer.pad_token_id,
    ).to(device)

    # Count trainable vs frozen params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable:,}  |  Frozen parameters: {frozen:,}")

    # Load dataset
    print(f"Loading {dataset_name.upper()} dataset...")
    dataset = load_dataset("glue", dataset_name)

    # Determine column names based on dataset
    if dataset_name == "mrpc":
        sent1_key, sent2_key = "sentence1", "sentence2"
    elif dataset_name == "qqp":
        sent1_key, sent2_key = "question1", "question2"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def tokenize(batch):
        sent1 = tokenizer(
            batch[sent1_key],
            truncation=True,
            padding="max_length",
            max_length=config["mask"]["max_tokens"],
        )
        sent2 = tokenizer(
            batch[sent2_key],
            truncation=True,
            padding="max_length",
            max_length=config["mask"]["max_tokens"],
        )
        return {
            "sent1_ids": sent1["input_ids"],
            "sent2_ids": sent2["input_ids"],
            "labels": batch["label"],
        }

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["sent1_ids", "sent2_ids", "labels"]
    )

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=batch_size
    )

    # Optimizer: head only (encoder is frozen)
    optimizer = torch.optim.AdamW([
        {'params': model.norm.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=lr, weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    print(f"Linear probing {dataset_name.upper()} with head lr={lr} (encoder is frozen)")

    # Training
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.encoder.eval()       # Always keep encoder in eval
        model.norm.train()
        model.classifier.train()

        correct = total = 0
        loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sent1_ids = batch["sent1_ids"].to(device)
            sent2_ids = batch["sent2_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(sent1_ids, sent2_ids)
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
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                sent1_ids = batch["sent1_ids"].to(device)
                sent2_ids = batch["sent2_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(sent1_ids, sent2_ids)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100.0 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_precision = precision_score(all_labels, all_preds, average='binary')
        val_recall = recall_score(all_labels, all_preds, average='binary')

        is_best = val_f1 > best_f1
        best_f1 = max(best_f1, val_f1)

        logger.log(
            epoch + 1,
            loss_sum / len(train_loader),
            train_acc,
            val_acc,
            val_f1,
            val_precision,
            val_recall,
            is_best,
        )

        print(
            f"Epoch {epoch+1}: Train={train_acc:.2f}% | "
            f"Val Acc={val_acc:.2f}% | F1={val_f1:.4f} | "
            f"P={val_precision:.4f} | R={val_recall:.4f}"
        )

    print(f"\nBEST F1 SCORE: {best_f1:.4f}")

    # Save model
    end_time = datetime.now()
    total_time = end_time - start_time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(
        output_dir, f"paraphrase_{dataset_name}_model_{timestamp}.pth"
    )

    # Only save the head (encoder is frozen, not needed)
    torch.save({
        'head_state_dict': {
            'norm': model.norm.state_dict(),
            'classifier': model.classifier.state_dict(),
        },
        'best_f1': best_f1,
        'num_classes': 2,
        'embed_dim': embed_dim,
        'config': config,
        'dataset': dataset_name,
    }, model_save_path)

    print(f"✓ Linear probe head saved to: {model_save_path}")

    # Save training log
    with open(logger.txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"PARAPHRASE DETECTION ({dataset_name.upper()}) LINEAR PROBING LOG\n")
        f.write("=" * 70 + "\n\n")

        f.write("EXPERIMENT INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Task: Paraphrase Detection\n")
        if dataset_name == "mrpc":
            f.write(f"Dataset: Microsoft Research Paraphrase Corpus (MRPC)\n")
        else:
            f.write(f"Dataset: Quora Question Pairs (QQP)\n")
        f.write(f"Classes: 2 (Not Paraphrase, Paraphrase)\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Training Time: {total_time}\n")
        f.write(f"Total Training Time (seconds): {total_time.total_seconds():.2f}s\n")
        f.write(f"Total Training Time (minutes): {total_time.total_seconds()/60:.2f}m\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Encoder Path: {encoder_path}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Embedding Dimension: {embed_dim}\n")
        f.write(f"Architecture: Dual-encoder with rich interactions (FROZEN)\n")
        f.write(f"  - Concatenation of both CLS embeddings\n")
        f.write(f"  - Element-wise difference\n")
        f.write(f"  - Element-wise product\n")
        f.write(f"Number of Classes: 2\n")
        f.write(f"Trainable Params: {trainable:,}\n")
        f.write(f"Frozen Params:    {frozen:,}\n\n")

        f.write("TRAINING HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Head Learning Rate: {lr}\n")
        f.write(f"Weight Decay: 0.01\n")
        f.write(f"Dropout: 0.2\n")
        f.write(f"Optimizer: AdamW (head only)\n")
        f.write(f"Loss Function: CrossEntropyLoss\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train Samples: {len(dataset['train'])}\n")
        f.write(f"Validation Samples: {len(dataset['validation'])}\n\n")

        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best F1 Score: {best_f1:.4f}\n\n")

        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write("SentencePairModel(\n")
        f.write("  Encoder x2 (Text-JEPA pretrained, FROZEN)\n")
        f.write("  LayerNorm                        ← trainable\n")
        f.write("  Dropout(0.2)                     ← trainable\n")
        f.write(f"  Linear({embed_dim * 4} -> 2)          ← trainable\n")
        f.write(")\n\n")

        f.write("NOTES\n")
        f.write("-" * 70 + "\n")
        f.write("- Encoder is FROZEN throughout training\n")
        f.write("- Only LayerNorm + Linear classifier are trained\n")
        f.write("- Encoder kept in eval() mode to disable dropout/batchnorm updates\n")
        f.write("- torch.no_grad() wraps encoder forward pass for efficiency\n")
        f.write("- CLS token used from both sentence encodings\n")
        f.write("- Rich interactions: [s1; s2; |s1-s2|; s1*s2]\n")
        f.write("- F1 score is the primary metric\n\n")

        f.write("=" * 70 + "\n")
        f.write("END OF TRAINING LOG\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Training log saved to: {logger.txt_path}")
    print(f"✓ Total training time: {total_time}")

    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Paraphrase Detection Linear Probing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mrpc",
                        choices=["mrpc", "qqp"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/paraphrase")

    args = parser.parse_args()

    finetune_paraphrase(
        encoder_path=args.checkpoint,
        config_path=args.config,
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )