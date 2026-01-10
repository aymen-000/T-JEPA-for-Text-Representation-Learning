"""
Embedding Visualization for Fine-tuned Text-JEPA Model

Visualizes learned embeddings using t-SNE and UMAP to assess
the model's ability to distinguish between classes.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer

from src.help.schedulers import init_model


# -------------------------------------------------------
# Fine-tuning Model (same as training)
# -------------------------------------------------------
class TextFineTuneModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, return_embeddings=False):
        feats = self.encoder(input_ids)
        cls_feat = feats[:, 0]
        
        cls_feat = self.norm(cls_feat)
        
        if return_embeddings:
            return cls_feat
        
        cls_feat = self.dropout(cls_feat)
        return self.classifier(cls_feat)


# -------------------------------------------------------
# Extract embeddings from model
# -------------------------------------------------------
def extract_embeddings(model, dataloader, device, max_samples=5000):
    """
    Extract embeddings and labels from the model.
    """
    model.eval()
    embeddings = []
    labels = []
    
    print(f"Extracting embeddings (max {max_samples} samples)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            
            # Get embeddings (before dropout and classifier)
            emb = model(input_ids, return_embeddings=True)
            
            embeddings.append(emb.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            
            # Stop if we have enough samples
            if len(embeddings) * dataloader.batch_size >= max_samples:
                break
    
    embeddings = np.vstack(embeddings)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]
    
    print(f"✓ Extracted {len(embeddings)} embeddings")
    return embeddings, labels


# -------------------------------------------------------
# Visualization functions
# -------------------------------------------------------
def plot_tsne(embeddings, labels, class_names, save_path, perplexity=30):
    """
    Create t-SNE visualization.
    """
    print(f"Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # Plot each class
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20,
            edgecolors='k',
            linewidths=0.5
        )
    
    plt.legend(fontsize=12, markerscale=2)
    plt.title("t-SNE Visualization of Learned Embeddings", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ t-SNE plot saved to: {save_path}")
    plt.close()


def plot_umap(embeddings, labels, class_names, save_path, n_neighbors=15):
    """
    Create UMAP visualization.
    """
    print(f"Computing UMAP (n_neighbors={n_neighbors})...")
    umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, n_jobs=-1)
    embeddings_2d = umap.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # Plot each class
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20,
            edgecolors='k',
            linewidths=0.5
        )
    
    plt.legend(fontsize=12, markerscale=2)
    plt.title("UMAP Visualization of Learned Embeddings", fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ UMAP plot saved to: {save_path}")
    plt.close()


def plot_density_heatmap(embeddings, labels, class_names, save_path, method='tsne'):
    """
    Create density heatmap for each class.
    """
    if method == 'tsne':
        print("Computing t-SNE for density plot...")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    else:
        print("Computing UMAP for density plot...")
        reducer = UMAP(n_components=2, n_neighbors=15, random_state=42, n_jobs=-1)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        
        axes[i].scatter(
            embeddings_2d[~mask, 0],
            embeddings_2d[~mask, 1],
            c='lightgray',
            alpha=0.3,
            s=10,
            label='Other classes'
        )
        
        axes[i].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=f'C{i}',
            alpha=0.7,
            s=20,
            edgecolors='k',
            linewidths=0.5,
            label=class_name
        )
        
        axes[i].set_title(f"{class_name}", fontsize=14, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    method_name = method.upper()
    fig.suptitle(f"{method_name} - Class Separation Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Density plot saved to: {save_path}")
    plt.close()


def compute_clustering_metrics(embeddings, labels):
    """
    Compute clustering quality metrics.
    """
    print("\nComputing clustering metrics...")
    
    # Silhouette Score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(embeddings, labels)
    
    # Davies-Bouldin Index (lower is better, >= 0)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    return silhouette, davies_bouldin


def plot_confusion_style_matrix(embeddings, labels, class_names, save_path):
    """
    Create a visualization showing inter-class distances.
    """
    from sklearn.metrics.pairwise import euclidean_distances
    
    n_classes = len(class_names)
    
    # Compute mean embedding for each class
    class_centers = []
    for i in range(n_classes):
        mask = labels == i
        class_centers.append(embeddings[mask].mean(axis=0))
    
    class_centers = np.array(class_centers)
    
    # Compute pairwise distances
    distances = euclidean_distances(class_centers)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distances,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd_r',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Euclidean Distance'}
    )
    plt.title("Inter-Class Distance Matrix\n(Higher = Better Separation)", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distance matrix saved to: {save_path}")
    plt.close()


def save_metrics_report(metrics, save_path):
    """
    Save metrics to a text file.
    """
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EMBEDDING QUALITY METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CLUSTERING METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Silhouette Score: {metrics['silhouette']:.4f}\n")
        f.write("  - Range: -1 (worst) to +1 (best)\n")
        f.write("  - Interpretation: Measures how similar samples are to their own cluster\n")
        f.write("  - > 0.5: Good separation\n")
        f.write("  - 0.25-0.5: Moderate separation\n")
        f.write("  - < 0.25: Poor separation\n\n")
        
        f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}\n")
        f.write("  - Range: 0 (best) to infinity (worst)\n")
        f.write("  - Interpretation: Ratio of within-cluster to between-cluster distances\n")
        f.write("  - < 1.0: Good separation\n")
        f.write("  - 1.0-2.0: Moderate separation\n")
        f.write("  - > 2.0: Poor separation\n\n")
        
        f.write("ANALYSIS\n")
        f.write("-" * 70 + "\n")
        
        if metrics['silhouette'] > 0.5 and metrics['davies_bouldin'] < 1.0:
            f.write("✓ EXCELLENT: Model shows strong ability to distinguish between classes\n")
        elif metrics['silhouette'] > 0.25 and metrics['davies_bouldin'] < 2.0:
            f.write("○ GOOD: Model shows reasonable separation between classes\n")
        else:
            f.write("✗ NEEDS IMPROVEMENT: Model shows limited class separation\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Metrics report saved to: {save_path}")


# -------------------------------------------------------
# Helper functions
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
# Main visualization function
# -------------------------------------------------------
def visualize_embeddings(
    model_path,
    config_path,
    model_name=None,
    batch_size=128,
    max_samples=5000,
    device="cuda",
    output_dir="outputs/visualizations",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(output_dir, f"viz_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load saved model
    print("Loading fine-tuned model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    embed_dim = checkpoint['embed_dim']
    num_classes = checkpoint['num_classes']
    config = checkpoint['config']
    
    # Load encoder checkpoint to get architecture details
    print("Loading config...")
    with open(config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    
    # Reconstruct encoder
    # We need the original encoder checkpoint path - assuming it's saved in config
    # For now, we'll extract from the saved model_state_dict
    
    # Get vocab size and other params from saved encoder state
    encoder_state = {k.replace('encoder.', ''): v 
                     for k, v in checkpoint['model_state_dict'].items() 
                     if k.startswith('encoder.')}
    
    vocab_size = encoder_state['token_embed.token_embed.weight'].shape[0]
    max_seq_len = encoder_state['pos_embed'].shape[1]
    
    depth = max(
        int(k.split(".")[1]) + 1
        for k in encoder_state
        if k.startswith("blocks.") and ".norm1.weight" in k
    )
    num_heads = 8
    
    if model_name is None:
        if vocab_size == 30522:
            model_name = "bert-base-uncased"
        elif vocab_size == 50257:
            model_name = "gpt2"
        elif vocab_size == 32000:
            model_name = "t5-base"
    
    print(f"Model: {model_name}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")
    
    # Initialize encoder
    encoder, _ = init_model(
        device=device,
        model_name=model_name,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        pred_depth=config_yaml["meta"]["pred_depth"],
        pred_emb_dim=config_yaml["meta"]["pred_emb_dim"],
    )
    
    # Reconstruct model
    model = TextFineTuneModel(encoder, embed_dim, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config_yaml["mask"]["max_tokens"],
        )
        out["labels"] = batch["label"]
        return out
    
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(model, test_loader, device, max_samples)
    
    # Compute metrics
    metrics = {}
    metrics['silhouette'], metrics['davies_bouldin'] = compute_clustering_metrics(
        embeddings, labels
    )
    
    # Save metrics report
    metrics_path = os.path.join(viz_dir, "metrics_report.txt")
    save_metrics_report(metrics, metrics_path)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. t-SNE
    tsne_path = os.path.join(viz_dir, "tsne_visualization.png")
    plot_tsne(embeddings, labels, class_names, tsne_path)
    
    # 2. UMAP
    umap_path = os.path.join(viz_dir, "umap_visualization.png")
    plot_umap(embeddings, labels, class_names, umap_path)
    
    # 3. Density plots (t-SNE)
    density_tsne_path = os.path.join(viz_dir, "class_separation_tsne.png")
    plot_density_heatmap(embeddings, labels, class_names, density_tsne_path, method='tsne')
    
    # 4. Density plots (UMAP)
    density_umap_path = os.path.join(viz_dir, "class_separation_umap.png")
    plot_density_heatmap(embeddings, labels, class_names, density_umap_path, method='umap')
    
    # 5. Distance matrix
    distance_path = os.path.join(viz_dir, "interclass_distance_matrix.png")
    plot_confusion_style_matrix(embeddings, labels, class_names, distance_path)
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: {viz_dir}")
    print(f"{'='*70}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Embedding Visualization")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to fine-tuned model .pth file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum samples to visualize")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations")
    
    args = parser.parse_args()
    
    visualize_embeddings(
        model_path=args.model,
        config_path=args.config,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
    )