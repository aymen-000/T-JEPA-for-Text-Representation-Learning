"""
Text-JEPA Fine-tuning on MNLI (Multi-Genre Natural Language Inference)

3-class classification: entailment, neutral, contradiction.
Encoder is FROZEN — only the MLP head is trained.

Dataset : GLUE MNLI  (matched split for validation)
Metric  : Accuracy (matched) — standard GLUE metric for MNLI
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
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Fine-tuning Model for Sentence-Pair Tasks
# -------------------------------------------------------
class SentencePairModel(nn.Module):
    """
    Dual encoder with rich interactions for sentence-pair classification.
    Identical pooling to TextLinearProbeModel: mean over non-padding tokens.
    Encoder is FROZEN — only norm + classifier are trained.
    """
    def __init__(self, encoder, embed_dim, num_classes, pad_id=0):
        super().__init__()
        self.encoder   = encoder
        self.pad_id    = pad_id
        self.norm      = nn.LayerNorm(embed_dim)
        self.dropout   = nn.Dropout(0.2)
        # [premise; hypothesis; |p-h|; p*h]  →  4 * D
        self.classifier = nn.Linear(embed_dim * 4, num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _mean_pool(self, input_ids: torch.Tensor,
                   feats: torch.Tensor) -> torch.Tensor:
        """Mean pool over non-padding tokens — matches TextLinearProbeModel."""
        mask = (input_ids != self.pad_id).unsqueeze(-1).float()  # (B, L, 1)
        return (feats * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, D)

    def forward(self, premise_ids, hypothesis_ids):
        with torch.no_grad():
            p_feats = self.encoder(premise_ids)     # (B, L, D)
            h_feats = self.encoder(hypothesis_ids)  # (B, L, D)

        p = self.norm(self._mean_pool(premise_ids,    p_feats))
        h = self.norm(self._mean_pool(hypothesis_ids, h_feats))

        diff     = torch.abs(p - h)
        prod     = p * h
        combined = torch.cat([p, h, diff, prod], dim=-1)
        return self.classifier(self.dropout(combined))


# -------------------------------------------------------
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    def __init__(self, output_dir="outputs/mnli"):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path  = os.path.join(output_dir, f"mnli_results_{self.timestamp}.csv")
        self.txt_path  = os.path.join(output_dir, f"mnli_log_{self.timestamp}.txt")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc",
                             "val_acc", "val_f1_macro", "is_best"])

        print(f"✓ CSV log : {self.csv_path}")
        print(f"✓ Text log: {self.txt_path}")

    def log(self, epoch, train_loss, train_acc, val_acc, val_f1, is_best):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_acc:.2f}", f"{val_f1:.4f}", int(is_best)])


def get_model_name_from_checkpoint(checkpoint):
    vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    if vocab_size == 30522: return "bert-base-uncased"
    if vocab_size == 50257: return "gpt2"
    if vocab_size == 32000: return "t5-base"
    return None


# -------------------------------------------------------
# Main fine-tuning function
# -------------------------------------------------------
def finetune_mnli(
    encoder_path,
    config_path,
    batch_size  = 32,
    num_epochs  = 15,
    lr          = 2e-5,
    device      = "cuda",
    output_dir  = "outputs/mnli",
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

    # MNLI: 3 classes — entailment(0), neutral(1), contradiction(2)
    model = SentencePairModel(
        encoder    = encoder,
        embed_dim  = embed_dim,
        num_classes= 3,
        pad_id     = tokenizer.pad_token_id or 0,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading MNLI …")
    dataset = load_dataset("glue", "mnli")

    max_len = config["mask"].get("max_tokens", 128)

    def tokenize(batch):
        p = tokenizer(batch["premise"],    truncation=True,
                      padding="max_length", max_length=max_len)
        h = tokenizer(batch["hypothesis"], truncation=True,
                      padding="max_length", max_length=max_len)
        return {
            "premise_ids":    p["input_ids"],
            "hypothesis_ids": h["input_ids"],
            "labels":         batch["label"],
        }

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch",
                       columns=["premise_ids", "hypothesis_ids", "labels"])

    train_loader = DataLoader(dataset["train"],             batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset["validation_matched"],batch_size=batch_size)

    # ── Optimiser: head only ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [{'params': model.norm.parameters()},
         {'params': model.classifier.parameters()}],
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Linear probing MNLI  |  lr={lr}  |  epochs={num_epochs}")

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.encoder.eval()
        model.norm.train()
        model.classifier.train()

        correct = total = 0
        loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            p_ids  = batch["premise_ids"].to(device)
            h_ids  = batch["hypothesis_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(p_ids, h_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds     = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        train_acc = 100.0 * correct / total

        # ── Validation (matched split) ────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                p_ids  = batch["premise_ids"].to(device)
                h_ids  = batch["hypothesis_ids"].to(device)
                labels = batch["labels"].to(device)

                preds  = model(p_ids, h_ids).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1  = f1_score(all_labels, all_preds, average="macro")
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        logger.log(epoch+1, loss_sum/len(train_loader),
                   train_acc, val_acc, val_f1, is_best)

        print(f"Epoch {epoch+1}: Train={train_acc:.2f}%  |  "
              f"Val Acc={val_acc:.2f}%  |  F1-macro={val_f1:.4f}"
              + ("  ← best" if is_best else ""))

    print(f"\nBEST VAL ACC: {best_acc:.2f}%")
    print(classification_report(all_labels, all_preds,
                                target_names=["entailment","neutral","contradiction"]))

    # ── Save head ─────────────────────────────────────────────────────────────
    end_time   = datetime.now()
    total_time = end_time - start_time
    timestamp  = end_time.strftime("%Y%m%d_%H%M%S")
    save_path  = os.path.join(output_dir, f"mnli_head_{timestamp}.pth")

    torch.save({
        "head_state_dict": {
            "norm":       model.norm.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
        "best_acc":   best_acc,
        "num_classes": 3,
        "embed_dim":   embed_dim,
        "config":      config,
        "task":        "mnli",
    }, save_path)
    print(f"✓ Head saved: {save_path}")

    # ── Text log ──────────────────────────────────────────────────────────────
    with open(logger.txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MNLI LINEAR PROBING LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Task              : Multi-Genre NLI (3-class)\n")
        f.write(f"Dataset           : GLUE MNLI (matched validation)\n")
        f.write(f"Classes           : entailment / neutral / contradiction\n")
        f.write(f"Start Time        : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time          : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time        : {total_time}\n\n")
        f.write(f"Encoder Path      : {encoder_path}\n")
        f.write(f"Embedding Dim     : {embed_dim}\n")
        f.write(f"Trainable Params  : {trainable:,}\n")
        f.write(f"Frozen Params     : {frozen:,}\n\n")
        f.write(f"Batch Size        : {batch_size}\n")
        f.write(f"Epochs            : {num_epochs}\n")
        f.write(f"LR                : {lr}\n\n")
        f.write(f"BEST VAL ACC      : {best_acc:.2f}%\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Log saved: {logger.txt_path}  |  Time: {total_time}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNLI Linear Probing")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--model_name",  type=str, default=None)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--output_dir",  type=str, default="outputs/mnli")
    args = parser.parse_args()

    finetune_mnli(
        encoder_path = args.checkpoint,
        config_path  = args.config,
        model_name   = args.model_name,
        batch_size   = args.batch_size,
        num_epochs   = args.epochs,
        lr           = args.lr,
        device       = args.device,
        output_dir   = args.output_dir,
    )