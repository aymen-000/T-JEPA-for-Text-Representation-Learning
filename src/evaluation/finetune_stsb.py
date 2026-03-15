"""
Text-JEPA Fine-tuning on STS-B (Semantic Textual Similarity Benchmark)

Regression task: predict similarity score in [0, 5] for a sentence pair.
This is the only regression task in GLUE — uses cosine similarity loss.

Encoder is FROZEN — only the regression head is trained.

Dataset : GLUE STS-B
Metrics : Pearson r and Spearman ρ correlation — both official GLUE metrics
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
from scipy.stats import pearsonr, spearmanr

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Regression Model for Sentence-Pair Similarity
# -------------------------------------------------------
class SentenceSimilarityModel(nn.Module):
    """
    Dual encoder → cosine similarity → scaled to [0, 5].
    Mean pool over non-padding tokens — matches TextLinearProbeModel exactly.
    Encoder is FROZEN — only norm + scalar scale are trained.

    Prediction: score = 2.5 * (1 + cosine_sim(u, v))
    This maps cosine ∈ [-1, 1] → score ∈ [0, 5] linearly.
    A learnable scale + bias further adjusts the range end-to-end.
    """
    def __init__(self, encoder, embed_dim, pad_id=0):
        super().__init__()
        self.encoder = encoder
        self.pad_id  = pad_id
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        # Learnable scale + bias on top of cosine similarity
        self.scale = nn.Parameter(torch.tensor(2.5))
        self.bias  = nn.Parameter(torch.tensor(2.5))

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _mean_pool(self, input_ids, feats):
        mask = (input_ids != self.pad_id).unsqueeze(-1).float()
        return (feats * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, sent1_ids, sent2_ids):
        with torch.no_grad():
            f1 = self.encoder(sent1_ids)   # (B, L, D)
            f2 = self.encoder(sent2_ids)   # (B, L, D)

        u = self.norm(self._mean_pool(sent1_ids, f1))   # (B, D)
        v = self.norm(self._mean_pool(sent2_ids, f2))   # (B, D)

        u = self.dropout(u)
        v = self.dropout(v)

        cos_sim = torch.cosine_similarity(u, v, dim=-1)         # (B,)
        # Map [-1,1] → [0,5] via learnable scale + bias
        return self.scale * cos_sim + self.bias                  # (B,)


# -------------------------------------------------------
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    def __init__(self, output_dir="outputs/stsb"):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path  = os.path.join(output_dir, f"stsb_results_{self.timestamp}.csv")
        self.txt_path  = os.path.join(output_dir, f"stsb_log_{self.timestamp}.txt")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_pearson",
                             "val_spearman", "val_mse", "is_best"])

        print(f"✓ CSV log : {self.csv_path}")
        print(f"✓ Text log: {self.txt_path}")

    def log(self, epoch, train_loss, val_pearson,
            val_spearman, val_mse, is_best):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_pearson:.4f}",
                             f"{val_spearman:.4f}", f"{val_mse:.4f}", int(is_best)])


def get_model_name_from_checkpoint(checkpoint):
    vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    if vocab_size == 30522: return "bert-base-uncased"
    if vocab_size == 50257: return "gpt2"
    if vocab_size == 32000: return "t5-base"
    return None


# -------------------------------------------------------
# Main fine-tuning function
# -------------------------------------------------------
def finetune_stsb(
    encoder_path,
    config_path,
    batch_size  = 32,
    num_epochs  = 15,
    lr          = 2e-5,
    device      = "cuda",
    output_dir  = "outputs/stsb",
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

    model = SentenceSimilarityModel(
        encoder  = encoder,
        embed_dim= embed_dim,
        pad_id   = tokenizer.pad_token_id or 0,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading STS-B …")
    dataset = load_dataset("glue", "stsb")

    max_len = config["mask"].get("max_tokens", 128)

    def tokenize(batch):
        s1 = tokenizer(batch["sentence1"], truncation=True,
                       padding="max_length", max_length=max_len)
        s2 = tokenizer(batch["sentence2"], truncation=True,
                       padding="max_length", max_length=max_len)
        return {
            "sent1_ids": s1["input_ids"],
            "sent2_ids": s2["input_ids"],
            # STS-B scores are floats in [0, 5]
            "labels":    batch["label"],
        }

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch",
                       columns=["sent1_ids", "sent2_ids", "labels"])

    train_loader = DataLoader(dataset["train"],      batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset["validation"], batch_size=batch_size)

    # ── Optimiser: head only ──────────────────────────────────────────────────
    # Only norm, scale, and bias have requires_grad=True
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()

    print(f"Linear probing STS-B  |  lr={lr}  |  epochs={num_epochs}")
    print("Primary metrics: Pearson r  &  Spearman ρ")

    best_spearman = -1.0

    for epoch in range(num_epochs):
        model.encoder.eval()
        model.norm.train()
        # scale and bias are Parameters so they update automatically

        loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            s1     = batch["sent1_ids"].to(device)
            s2     = batch["sent2_ids"].to(device)
            labels = batch["labels"].float().to(device)    # float for MSE

            optimizer.zero_grad()
            preds = model(s1, s2)                          # (B,)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                s1     = batch["sent1_ids"].to(device)
                s2     = batch["sent2_ids"].to(device)
                labels = batch["labels"].float().to(device)
                preds  = model(s1, s2)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_pearson,  _ = pearsonr(all_labels,  all_preds)
        val_spearman, _ = spearmanr(all_labels, all_preds)
        val_mse         = sum((p - l) ** 2 for p, l in zip(all_preds, all_labels)) / len(all_labels)

        is_best      = val_spearman > best_spearman
        best_spearman = max(best_spearman, val_spearman)

        logger.log(epoch+1, loss_sum/len(train_loader),
                   val_pearson, val_spearman, val_mse, is_best)

        print(f"Epoch {epoch+1}: Loss={loss_sum/len(train_loader):.4f}  |  "
              f"Pearson={val_pearson:.4f}  |  Spearman={val_spearman:.4f}  |  "
              f"MSE={val_mse:.4f}"
              + ("  ← best" if is_best else ""))

    print(f"\nBEST SPEARMAN ρ : {best_spearman:.4f}")

    # ── Save head ─────────────────────────────────────────────────────────────
    end_time   = datetime.now()
    total_time = end_time - start_time
    timestamp  = end_time.strftime("%Y%m%d_%H%M%S")
    save_path  = os.path.join(output_dir, f"stsb_head_{timestamp}.pth")

    torch.save({
        "head_state_dict": {
            "norm":  model.norm.state_dict(),
            "scale": model.scale.data,
            "bias":  model.bias.data,
        },
        "best_spearman": best_spearman,
        "embed_dim":     embed_dim,
        "config":        config,
        "task":          "stsb",
    }, save_path)
    print(f"✓ Head saved: {save_path}")

    # ── Text log ──────────────────────────────────────────────────────────────
    with open(logger.txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STS-B LINEAR PROBING LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Task              : Semantic Textual Similarity (regression)\n")
        f.write(f"Dataset           : GLUE STS-B  (scores 0–5)\n")
        f.write(f"Primary Metrics   : Pearson r  &  Spearman ρ\n")
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
        f.write(f"Loss              : MSELoss\n")
        f.write(f"Architecture      : cosine_sim(u,v) → scale*x + bias → [0,5]\n\n")
        f.write(f"BEST SPEARMAN ρ   : {best_spearman:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Log saved: {logger.txt_path}  |  Time: {total_time}")
    return best_spearman


if __name__ == "__main__":
    parser = argparse.ArgumentParser("STS-B Linear Probing")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--model_name",  type=str, default=None)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--output_dir",  type=str, default="outputs/stsb")
    args = parser.parse_args()

    finetune_stsb(
        encoder_path = args.checkpoint,
        config_path  = args.config,
        model_name   = args.model_name,
        batch_size   = args.batch_size,
        num_epochs   = args.epochs,
        lr           = args.lr,
        device       = args.device,
        output_dir   = args.output_dir,
    )