"""
Text-JEPA Fine-tuning on CoLA (Corpus of Linguistic Acceptability)

Binary classification: is a sentence grammatically acceptable?
Single-sentence task — no sentence pair, simpler than MRPC/MNLI.

Encoder is FROZEN — only the MLP head is trained.

Dataset : GLUE CoLA
Metric  : Matthews Correlation Coefficient (MCC) — official GLUE metric for CoLA
          Accuracy reported alongside for reference.
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
from sklearn.metrics import (matthews_corrcoef, f1_score,
                             precision_score, recall_score)

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Linear Probe Model  (single sentence)
# -------------------------------------------------------
class TextLinearProbeModel(nn.Module):
    """
    Single-sentence linear probe.
    Mean pool over non-padding tokens → LayerNorm → Dropout → Linear.
    Encoder is FROZEN — only norm + classifier are trained.
    """
    def __init__(self, encoder, embed_dim, num_classes, pad_id=0):
        super().__init__()
        self.encoder    = encoder
        self.pad_id     = pad_id
        self.norm       = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Linear(embed_dim, num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        with torch.no_grad():
            feats = self.encoder(input_ids)                         # (B, L, D)
        mask      = (input_ids != self.pad_id).unsqueeze(-1).float()
        sent_feat = (feats * mask).sum(1) / mask.sum(1).clamp(min=1)
        sent_feat = self.norm(sent_feat)
        sent_feat = self.dropout(sent_feat)
        return self.classifier(sent_feat)


# -------------------------------------------------------
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    def __init__(self, output_dir="outputs/cola"):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path  = os.path.join(output_dir, f"cola_results_{self.timestamp}.csv")
        self.txt_path  = os.path.join(output_dir, f"cola_log_{self.timestamp}.txt")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc",
                             "val_acc", "val_mcc", "val_f1", "is_best"])

        print(f"✓ CSV log : {self.csv_path}")
        print(f"✓ Text log: {self.txt_path}")

    def log(self, epoch, train_loss, train_acc,
            val_acc, val_mcc, val_f1, is_best):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_acc:.2f}", f"{val_mcc:.4f}",
                             f"{val_f1:.4f}", int(is_best)])


def get_model_name_from_checkpoint(checkpoint):
    vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    if vocab_size == 30522: return "bert-base-uncased"
    if vocab_size == 50257: return "gpt2"
    if vocab_size == 32000: return "t5-base"
    return None


# -------------------------------------------------------
# Main fine-tuning function
# -------------------------------------------------------
def finetune_cola(
    encoder_path,
    config_path,
    batch_size  = 32,
    num_epochs  = 15,
    lr          = 2e-5,
    device      = "cuda",
    output_dir  = "outputs/cola",
    model_name  = None,
):
    device     = torch.device(device if torch.cuda.is_available() else "cpu")
    start_time = datetime.now()
    logger     = CSVLogger(output_dir)
    print(f"Device: {device}")

    # ── Config + checkpoint ───────────────────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint  = torch.load(encoder_path, map_location=device)
    vocab_size  = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    embed_dim   = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[1]
    max_seq_len = checkpoint["encoder"]["pos_embed"].shape[1]
    depth       = max(
        int(k.split(".")[1]) + 1
        for k in checkpoint["encoder"]
        if k.startswith("blocks.") and ".norm1.weight" in k
    )

    if model_name is None:
        model_name = get_model_name_from_checkpoint(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder, _ = init_model(
        device       = device,
        model_name   = model_name,
        vocab_size   = vocab_size,
        max_seq_len  = max_seq_len,
        embed_dim    = embed_dim,
        depth        = depth,
        num_heads    = 8,
        pred_depth   = config["meta"]["pred_depth"],
        pred_emb_dim = config["meta"]["pred_emb_dim"],
    )
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    # CoLA: 2 classes — unacceptable(0), acceptable(1)
    model = TextLinearProbeModel(
        encoder     = encoder,
        embed_dim   = embed_dim,
        num_classes = 2,
        pad_id      = tokenizer.pad_token_id or 0,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # CoLA is class-imbalanced (~70% acceptable) so we use weighted loss
    print("Loading CoLA …")
    dataset = load_dataset("glue", "cola")

    max_len = config["mask"].get("max_tokens", 128)

    def tokenize(batch):
        enc = tokenizer(batch["sentence"], truncation=True,
                        padding="max_length", max_length=max_len)
        return {"input_ids": enc["input_ids"], "labels": batch["label"]}

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Class weights to handle imbalance (~30% unacceptable, ~70% acceptable)
    train_labels = dataset["train"]["labels"].tolist()
    n_total  = len(train_labels)
    n_pos    = sum(train_labels)
    n_neg    = n_total - n_pos
    # weight for class c = n_total / (n_classes * n_c)
    weights  = torch.tensor([n_total / (2 * n_neg),
                              n_total / (2 * n_pos)],
                             dtype=torch.float, device=device)
    print(f"Class weights  unacceptable={weights[0]:.3f}  acceptable={weights[1]:.3f}")

    train_loader = DataLoader(dataset["train"],      batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset["validation"], batch_size=batch_size)

    # ── Optimiser: head only ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [{'params': model.norm.parameters()},
         {'params': model.classifier.parameters()}],
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"Linear probing CoLA  |  lr={lr}  |  epochs={num_epochs}")
    print("Primary metric: MCC (Matthews Correlation Coefficient)")

    best_mcc = -1.0

    for epoch in range(num_epochs):
        model.encoder.eval()
        model.norm.train()
        model.classifier.train()

        correct = total = 0
        loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds     = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        train_acc = 100.0 * correct / total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                preds     = model(input_ids).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_mcc = matthews_corrcoef(all_labels, all_preds)   # PRIMARY METRIC
        val_f1  = f1_score(all_labels, all_preds, average="binary")

        is_best  = val_mcc > best_mcc
        best_mcc = max(best_mcc, val_mcc)

        logger.log(epoch+1, loss_sum/len(train_loader),
                   train_acc, val_acc, val_mcc, val_f1, is_best)

        print(f"Epoch {epoch+1}: Train={train_acc:.2f}%  |  "
              f"Val Acc={val_acc:.2f}%  |  MCC={val_mcc:.4f}  |  F1={val_f1:.4f}"
              + ("  ← best" if is_best else ""))

    print(f"\nBEST MCC: {best_mcc:.4f}")

    # ── Save head ─────────────────────────────────────────────────────────────
    end_time   = datetime.now()
    total_time = end_time - start_time
    timestamp  = end_time.strftime("%Y%m%d_%H%M%S")
    save_path  = os.path.join(output_dir, f"cola_head_{timestamp}.pth")

    torch.save({
        "head_state_dict": {
            "norm":       model.norm.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
        "best_mcc":    best_mcc,
        "num_classes": 2,
        "embed_dim":   embed_dim,
        "config":      config,
        "task":        "cola",
    }, save_path)
    print(f"✓ Head saved: {save_path}")

    # ── Text log ──────────────────────────────────────────────────────────────
    with open(logger.txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CoLA LINEAR PROBING LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Task              : Linguistic Acceptability (binary)\n")
        f.write(f"Dataset           : GLUE CoLA\n")
        f.write(f"Classes           : unacceptable / acceptable\n")
        f.write(f"Primary Metric    : Matthews Correlation Coefficient (MCC)\n")
        f.write(f"Start Time        : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time          : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time        : {total_time}\n\n")
        f.write(f"Encoder Path      : {encoder_path}\n")
        f.write(f"Embedding Dim     : {embed_dim}\n")
        f.write(f"Trainable Params  : {trainable:,}\n")
        f.write(f"Frozen Params     : {frozen:,}\n\n")
        f.write(f"Batch Size        : {batch_size}\n")
        f.write(f"Epochs            : {num_epochs}\n")
        f.write(f"LR                : {lr}\n")
        f.write(f"Class Weights     : unacceptable={weights[0].item():.3f}  "
                f"acceptable={weights[1].item():.3f}\n\n")
        f.write(f"BEST MCC          : {best_mcc:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Log saved: {logger.txt_path}  |  Time: {total_time}")
    return best_mcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CoLA Linear Probing")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--model_name",  type=str, default=None)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--output_dir",  type=str, default="outputs/cola")
    args = parser.parse_args()

    finetune_cola(
        encoder_path = args.checkpoint,
        config_path  = args.config,
        model_name   = args.model_name,
        batch_size   = args.batch_size,
        num_epochs   = args.epochs,
        lr           = args.lr,
        device       = args.device,
        output_dir   = args.output_dir,
    )