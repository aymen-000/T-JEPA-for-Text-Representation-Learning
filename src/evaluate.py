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
import csv
import os
from datetime import datetime

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
# CSV Logger
# -------------------------------------------------------
class CSVLogger:
    """Logger for saving training metrics to CSV"""
    
    def __init__(self, output_dir="outputs/text_jepa"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"linear_probe_results_{timestamp}.csv")
        
        # Initialize CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'is_best'])
        
        print(f"✓ CSV log created at: {self.csv_path}")
    
    def log(self, epoch, train_loss, train_acc, val_acc, is_best):
        """Append a row to the CSV file"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", 
                           f"{val_acc:.2f}", int(is_best)])
    
    def save_summary(self, best_acc, total_epochs, config_info):
        """Save a summary file with final results"""
        summary_path = self.csv_path.replace('.csv', '_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TEXT-JEPA LINEAR PROBING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Epochs: {total_epochs}\n")
            f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n\n")
            f.write("Configuration:\n")
            for key, value in config_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Summary saved at: {summary_path}")


# -------------------------------------------------------
# Helper function to detect model name from checkpoint
# -------------------------------------------------------
def get_model_name_from_checkpoint(checkpoint):
    """
    Detect the correct model name from checkpoint vocab size
    """
    if "encoder" in checkpoint:
        vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
        
        # Map vocab sizes to model names
        if vocab_size == 30522:
            return "bert-base-uncased"
        elif vocab_size == 50257:
            return "gpt2"
        elif vocab_size == 32000:
            return "t5-base"
        else:
            print(f"Warning: Unknown vocab size {vocab_size}")
            return None
    return None


# -------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------
def evaluate_linear_probe(
    encoder_path,
    config_path,
    model_name=None,  # Now optional
    num_classes=4,
    batch_size=64,
    num_epochs=20,
    lr=1e-3,
    device="cuda",
    output_dir="outputs/text_jepa",
):

    print("=" * 80)
    print("TEXT-JEPA LINEAR PROBING (AG NEWS)")
    print("=" * 80)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------
    # Initialize CSV Logger
    # ---------------------------------------------------
    logger = CSVLogger(output_dir=output_dir)

    # ---------------------------------------------------
    # Load config and checkpoint first to get actual dimensions
    # ---------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---------------------------------------------------
    # Load checkpoint and detect architecture parameters
    # ---------------------------------------------------
    print(f"\nLoading encoder from: {encoder_path}")
    checkpoint = torch.load(encoder_path, map_location=device)
    
    # Extract architecture details from checkpoint
    checkpoint_vocab_size = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[0]
    checkpoint_embed_dim = checkpoint["encoder"]["token_embed.token_embed.weight"].shape[1]
    checkpoint_max_seq_len = checkpoint["encoder"]["pos_embed"].shape[1]
    
    # Try to get encoder depth from checkpoint
    checkpoint_depth = 0
    for key in checkpoint["encoder"].keys():
        if key.startswith("blocks.") and ".norm1.weight" in key:
            block_num = int(key.split(".")[1])
            checkpoint_depth = max(checkpoint_depth, block_num + 1)
    
    # Try to get num_heads from attention layer
    # qkv.weight shape is [3*embed_dim*num_heads/num_heads, embed_dim] = [3*embed_dim, embed_dim]
    # So qkv.weight.shape[0] = 3 * embed_dim
    if "blocks.0.attn.qkv.weight" in checkpoint["encoder"]:
        qkv_out = checkpoint["encoder"]["blocks.0.attn.qkv.weight"].shape[0]
        # qkv_out = 3 * embed_dim, so we can verify
        expected_qkv = 3 * checkpoint_embed_dim
        if qkv_out == expected_qkv:
            # Default assumption: embed_dim must be divisible by num_heads
            # Common values: 8, 12, 16 heads
            for num_heads in [8, 12, 16, 4]:
                if checkpoint_embed_dim % num_heads == 0:
                    checkpoint_num_heads = num_heads
                    break
            else:
                checkpoint_num_heads = 8  # fallback
        else:
            checkpoint_num_heads = 8
    else:
        checkpoint_num_heads = 8
    
    print(f"\nCheckpoint architecture:")
    print(f"  Vocab size:    {checkpoint_vocab_size}")
    print(f"  Embed dim:     {checkpoint_embed_dim}")
    print(f"  Max seq len:   {checkpoint_max_seq_len}")
    print(f"  Encoder depth: {checkpoint_depth}")
    print(f"  Num heads:     {checkpoint_num_heads} (estimated)")
    
    # Get predictor embed_dim (this is what pred_emb_dim means)
    pred_emb_dim = config["meta"]["pred_emb_dim"]
    pred_depth = config["meta"]["pred_depth"]
    
    print(f"\nPredictor architecture:")
    print(f"  Pred embed dim: {pred_emb_dim}")
    print(f"  Pred depth:     {pred_depth}")

    # ---------------------------------------------------
    # Detect model name from vocab size for tokenizer
    # ---------------------------------------------------
    if model_name is None:
        if "config" in checkpoint and "model_name" in checkpoint["config"]:
            model_name = checkpoint["config"]["model_name"]
            print(f"\nUsing model name from checkpoint: {model_name}")
        else:
            detected_name = get_model_name_from_checkpoint(checkpoint)
            if detected_name:
                model_name = detected_name
                print(f"\nDetected tokenizer model: {model_name}")
            else:
                model_name = config["meta"]["model_name"]
                print(f"\nUsing model name from config: {model_name}")
    else:
        print(f"\nUsing explicitly specified model: {model_name}")
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_vocab_size = len(tokenizer)
    
    print(f"\nTokenizer check:")
    print(f"  Checkpoint vocab: {checkpoint_vocab_size}")
    print(f"  Tokenizer vocab:  {tokenizer_vocab_size}")
    
    if checkpoint_vocab_size != tokenizer_vocab_size:
        print(f"⚠️  WARNING: Vocab size mismatch!")
        print(f"   Using tokenizer with vocab size {tokenizer_vocab_size}")
        print(f"   but checkpoint was trained with {checkpoint_vocab_size}")
        # Try to find correct tokenizer
        if checkpoint_vocab_size == 30522:
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"   Switching to bert-base-uncased tokenizer")
        elif checkpoint_vocab_size == 50257:
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"   Switching to gpt2 tokenizer")
    else:
        print("✓ Vocab sizes match!")
    # ---------------------------------------------------
    # Initialize encoder with correct architecture from checkpoint
    # ---------------------------------------------------
    print(f"\nInitializing encoder with checkpoint architecture...")
    encoder, _ = init_model(
        device=device,
        model_name=model_name,  # Only used for API compatibility
        vocab_size=checkpoint_vocab_size,
        max_seq_len=checkpoint_max_seq_len,
        embed_dim=checkpoint_embed_dim,  # Use checkpoint's embed_dim!
        depth=checkpoint_depth,
        num_heads=checkpoint_num_heads,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
    )
    
    # Verify the encoder was created with correct dimensions
    encoder_vocab_size = encoder.token_embed.token_embed.weight.shape[0]
    encoder_embed_dim = encoder.token_embed.token_embed.weight.shape[1]
    
    print(f"\nEncoder verification:")
    print(f"  Created vocab size: {encoder_vocab_size}")
    print(f"  Created embed dim:  {encoder_embed_dim}")
    
    if encoder_vocab_size != checkpoint_vocab_size or encoder_embed_dim != checkpoint_embed_dim:
        raise ValueError(
            f"Failed to create encoder with correct architecture!\n"
            f"Expected: vocab={checkpoint_vocab_size}, embed={checkpoint_embed_dim}\n"
            f"Got:      vocab={encoder_vocab_size}, embed={encoder_embed_dim}"
        )
    
    print("✓ Encoder architecture matches checkpoint!")
    
    # Load the weights
    print("\nLoading checkpoint weights...")
    encoder.load_state_dict(checkpoint["encoder"])
    print("✓ Weights loaded successfully!")
    
    encoder.eval()

    # ---------------------------------------------------
    # Linear Probe
    # ---------------------------------------------------
    model = TextLinearProbe(
        encoder=encoder,
        embed_dim=checkpoint_embed_dim,  # Use actual encoder embed_dim
        num_classes=num_classes,
    ).to(device)

    # ---------------------------------------------------
    # Dataset (AG News)
    # ---------------------------------------------------
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

        train_loss = loss_sum / len(train_loader)
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
        is_best = val_acc > best_acc
        
        if is_best:
            best_acc = val_acc
        
        # Log to CSV
        logger.log(epoch + 1, train_loss, train_acc, val_acc, is_best)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")
        
        if is_best:
            print("✓ New best accuracy")

    print("=" * 80)
    print(f"BEST LINEAR PROBE ACCURACY: {best_acc:.2f}%")
    print("=" * 80)

    # Save summary
    config_info = {
        "encoder_path": encoder_path,
        "config_path": config_path,
        "model_name": model_name,
        "vocab_size": checkpoint_vocab_size,
        "embed_dim": checkpoint_embed_dim,
        "encoder_depth": checkpoint_depth,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "num_classes": num_classes,
    }
    logger.save_summary(best_acc, num_epochs, config_info)

    return best_acc


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Text-JEPA Linear Probing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (e.g., bert-base-uncased, gpt2). If not specified, will auto-detect.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/text_jepa")

    args = parser.parse_args()

    evaluate_linear_probe(
        encoder_path=args.checkpoint,
        config_path=args.config,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )