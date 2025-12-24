"""
Text-JEPA Linear Probing Evaluation

Evaluates a pretrained Text-JEPA encoder using
LINEAR PROBING on AG News.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Linear Probe Model
# -------------------------------------------------------
class TextLinearProbe(nn.Module):
    """
    Linear classifier on top of a frozen Text-JEPA encoder
    """
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, tokens):
        """
        tokens: [B, L]
        """
        with torch.no_grad():
            feats = self.encoder(tokens)     # [B, L, D]
            feats = feats.mean(dim=1)        # mean pooling over tokens

        return self.classifier(feats)


# -------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------
def evaluate_linear_probe(
    encoder_path,
    config_path,
    model_name="bert-base-uncased",
    num_classes=4,
    batch_size=64,
    num_epochs=20,
    lr=1e-3,
    device="cuda",
):

    print("=" * 80)
    print("TEXT-JEPA LINEAR PROBING (AG NEWS)")
    print("=" * 80)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------
    # Load config
    # ---------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    embed_dim = config["meta"]["pred_emb_dim"]

    # ---------------------------------------------------
    # Load encoder
    # ---------------------------------------------------
    print(f"\nLoading encoder from: {encoder_path}")
    checkpoint = torch.load(encoder_path, map_location=device)

    encoder, _ = init_model(
        device=device,
        model_name=config["meta"]["model_name"],
        pred_depth=config["meta"]["pred_depth"],
        pred_emb_dim=embed_dim,
    )

    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    # ---------------------------------------------------
    # Linear Probe
    # ---------------------------------------------------
    model = TextLinearProbe(
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=num_classes,
    ).to(device)

    # ---------------------------------------------------
    # Dataset (AG News)
    # ---------------------------------------------------
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------
    # Optimizer
    # ---------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # Training Loop
    # ---------------------------------------------------
    best_acc = 0.0
    print("\nTraining linear probe...")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        # ---------------- Validation ----------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(tokens)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            print("✓ New best accuracy")

    print("=" * 80)
    print(f"BEST LINEAR PROBE ACCURACY: {best_acc:.2f}%")
    print("=" * 80)

    return best_acc


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Text-JEPA Linear Probing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluate_linear_probe(
        encoder_path=args.checkpoint,
        config_path=args.config,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
