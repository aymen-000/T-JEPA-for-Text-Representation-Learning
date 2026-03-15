"""
Representation Analysis for Text-JEPA vs BERT (MLM-only)
=========================================================
Investigates encoder representation quality across 4 axes:

  1. Eigenvalue Spectrum       — how "spread" the representation space is
  2. Effective Rank            — dimensionality actually used by the encoder
  3. Token vs Semantic Probes  — does the encoder encode surface form or meaning?
  4. Alignment & Uniformity    — geometric quality of the embedding hypersphere

Datasets  : SST-2 (sentiment), MRPC / Microsoft Paraphrase (paraphrase)
Inspired by:
  - Wang & Isola (2020)  "Understanding Contrastive Representation Learning
                          through Alignment and Uniformity on the Hypersphere"
  - Roy & Vetrov (2007)  "The Effective Rank: A Measure of Effective Dimensionality"
  - Ethayarajh (2019)    "How Contextual are Contextualized Word Representations?"
  - Garrido et al.(2023) "On the Duality Between Contrastive and Non-Contrastive
                          Self-Supervised Learning" (eigenspectrum analysis)

"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (research-paper style)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'text_jepa': '#E63946',   # vivid red
    'bert':      '#457B9D',   # steel blue
    'neutral':   '#6B7280',
}
LABELS = {
    'text_jepa': 'Text-JEPA',
    'bert':      'BERT (MLM)',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linewidth':   0.6,
    'figure.dpi':       150,
})


def _load_encoder_from_ckpt(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Loads the encoder from a checkpoint saved .
    Reads model hyperparameters from the embedded 'config' dict 
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    expected_keys = {'encoder', 'config', 'training_mode'}
    if not expected_keys.issubset(ckpt.keys()):
        raise ValueError(
            f"Checkpoint at '{ckpt_path}' is missing keys. "
            f"Expected at least {expected_keys}, got {set(ckpt.keys())}."
        )

    cfg  = ckpt['config']
    mode = ckpt.get('training_mode', 'unknown')
    print(f"    checkpoint training_mode = '{mode}'")

    from src.help.schedulers import init_model
    encoder, _ = init_model(
        device=device,
        model_name  = cfg['meta']['model_name'],
        pred_depth  = cfg['meta']['pred_depth'],
        pred_emb_dim= cfg['meta']['pred_emb_dim'],
        vocab_size  = cfg['data'].get('vocab_size',  30522),
        max_seq_len = cfg['data'].get('max_seq_len', 512),
    )
    encoder.load_state_dict(ckpt['encoder'])
    encoder.to(device).eval()
    return encoder


class EncoderWrapper(nn.Module):
    """
    Unified wrapper for both the Text-JEPA and the MLM-baseline encoder.

    Pooling: mean over non-padding tokens, exactly as in TextLinearProbeModel:
        feats = encoder(input_ids)                      # (B, L, D)
        mask  = (input_ids != pad_token_id)             # (B, L, 1)
        sent  = (feats * mask).sum(1) / mask.sum(1)     # (B, D)
    """
    def __init__(self, ckpt_path: str, device: torch.device,
                 pad_token_id: int = 0):
        super().__init__()
        print(f"  Loading encoder from: {ckpt_path}")
        self.encoder      = _load_encoder_from_ckpt(ckpt_path, device)
        self.pad_token_id = pad_token_id
        self.device       = device

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns mean-pooled sentence embedding (B, D)."""
        with torch.no_grad():
            out = self.encoder(input_ids)                                   # (B, L, D)
        mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()       # (B, L, 1)
        out  = out * mask
        return out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)               # (B, D)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DATASET HELPERS
# ═════════════════════════════════════════════════════════════════════════════

DATASET_CFG = {
    'sst2': {
        'hf_path':     ('glue', 'sst2'),
        'text_col':    'sentence',
        'label_col':   'label',
        'task':        'classification',
        'n_classes':   2,
        'label_names': ['negative', 'positive'],
    },
    'mrpc': {
        'hf_path':     ('glue', 'mrpc'),
        'text_col':    ('sentence1', 'sentence2'),
        'label_col':   'label',
        'task':        'paraphrase',
        'n_classes':   2,
        'label_names': ['not-paraphrase', 'paraphrase'],
    },
}


def load_split(dataset_name: str, split: str, tokenizer,
               max_length: int, max_samples: int, device: torch.device):
    """
    Returns:
        input_ids     : (N, max_length)
        attention_mask: (N, max_length)
        labels        : (N,)   int labels
        raw_texts     : list[str]
    """
    cfg = DATASET_CFG[dataset_name]
    path = cfg['hf_path']
    ds   = load_dataset(*path, split=split)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    if isinstance(cfg['text_col'], tuple):
        texts = [f"{row[cfg['text_col'][0]]} [SEP] {row[cfg['text_col'][1]]}"
                 for row in ds]
    else:
        texts = [row[cfg['text_col']] for row in ds]

    labels = [row[cfg['label_col']] for row in ds]

    enc = tokenizer(texts, max_length=max_length, padding='max_length',
                    truncation=True, return_tensors='pt')

    return (enc['input_ids'].to(device),
            enc['attention_mask'].to(device),
            torch.tensor(labels, dtype=torch.long).to(device),
            texts)


@torch.no_grad()
def extract_embeddings(model, input_ids, attention_mask,
                       batch_size=64) -> np.ndarray:
    """Encode in mini-batches, return numpy (N, D)."""
    all_embs = []
    N = input_ids.size(0)
    for i in range(0, N, batch_size):
        ids  = input_ids[i:i+batch_size]
        mask = attention_mask[i:i+batch_size]
        emb  = model.encode(ids, mask)
        all_embs.append(emb.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ANALYSIS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 3.1  Eigenvalue Spectrum ──────────────────────────────────────────────────

def compute_eigenspectrum(Z: np.ndarray) -> dict:
    """
    Compute the singular-value (eigenvalue) spectrum of the centred embedding
    matrix Z ∈ R^{N×D}.

    Returns dict with:
        eigenvalues      : sorted descending singular values
        explained_var    : fraction of total variance per component
        cumulative_var   : cumulative explained variance
        spectral_entropy : normalised Shannon entropy of the eigenvalue dist.
                           (1 = perfectly uniform = richest representation)
    """
    Z_c = Z - Z.mean(axis=0, keepdims=True)         # centre
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
    eigvals  = S ** 2
    total    = eigvals.sum()
    exp_var  = eigvals / (total + 1e-12)
    cum_var  = np.cumsum(exp_var)

    # Spectral entropy (Garrido et al. 2023 metric)
    p = exp_var / (exp_var.sum() + 1e-12)
    spec_entropy = scipy_entropy(p) / np.log(len(p) + 1e-12)

    return {
        'eigenvalues':      eigvals,
        'explained_var':    exp_var,
        'cumulative_var':   cum_var,
        'spectral_entropy': float(spec_entropy),
    }


# ── 3.2  Effective Rank ───────────────────────────────────────────────────────

def effective_rank(Z: np.ndarray) -> dict:
    """
    Three complementary effective-rank measures:

    1. Roy & Vetrov (2007) erank:
           erank(A) = exp( H( σ / ‖σ‖₁ ) )
       where H is Shannon entropy of the normalised singular-value distribution.

    2. 90% variance rank:  smallest k s.t. ∑_{i≤k} λᵢ / ∑λ ≥ 0.90

    3. Stable rank (Vershynin 2018):
           srank(A) = ‖A‖_F² / ‖A‖₂²
    """
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)

    # 1. Roy & Vetrov erank
    p    = S / (S.sum() + 1e-12)
    p    = p[p > 1e-12]
    erank = float(np.exp(-np.sum(p * np.log(p))))

    # 2. 90%-variance rank
    eigvals  = S ** 2
    cum_var  = np.cumsum(eigvals) / (eigvals.sum() + 1e-12)
    rank_90  = int(np.searchsorted(cum_var, 0.90) + 1)

    # 3. Stable rank
    stable_rank = float((S ** 2).sum() / (S[0] ** 2 + 1e-12))

    return {
        'erank':        erank,
        'rank_90':      rank_90,
        'stable_rank':  stable_rank,
        'singular_vals': S,
    }


# ── 3.3  Token vs Semantic Probes ────────────────────────────────────────────

def token_probe(Z: np.ndarray, texts: list[str],
                tokenizer) -> dict:
    """
    Surface-form probe: predict the most-frequent token in each sentence
    directly from the mean-pooled sentence embedding.

    A *lower* probe accuracy means the representation does NOT encode
    surface tokens → richer semantic abstraction.
    Inspired by Conneau et al. (2018) SentEval probing tasks.
    """
    # Most-frequent non-special token id per sentence
    special_ids = set(tokenizer.all_special_ids)
    targets = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        ids = [i for i in ids if i not in special_ids]
        if ids:
            targets.append(int(np.bincount(ids).argmax()))
        else:
            targets.append(0)

    targets = np.array(targets)
    # Keep only classes with ≥5 samples to avoid degenerate CV
    classes, counts = np.unique(targets, return_counts=True)
    valid_cls = classes[counts >= 5]
    mask = np.isin(targets, valid_cls)
    Z_sub, y_sub = Z[mask], targets[mask]

    if len(np.unique(y_sub)) < 2:
        return {'token_probe_acc': float('nan'), 'n_classes': 0}

    scaler = StandardScaler()
    Z_s    = scaler.fit_transform(Z_sub)
    clf    = LogisticRegression(max_iter=500, C=1.0, solver='saga',
                                multi_class='auto', n_jobs=-1)
    scores = cross_val_score(clf, Z_s, y_sub, cv=3,
                             scoring='accuracy', n_jobs=-1)
    return {
        'token_probe_acc': float(scores.mean()),
        'token_probe_std': float(scores.std()),
        'n_classes':        int(len(np.unique(y_sub))),
    }


def semantic_probe(Z: np.ndarray, labels: np.ndarray,
                   label_names: list[str]) -> dict:
    """
    Semantic probe: predict the downstream label (sentiment / paraphrase)
    from frozen mean-pooled embeddings via linear classifier.

    Higher accuracy → encoder captures task-relevant semantics.
    """
    scaler = StandardScaler()
    Z_s    = scaler.fit_transform(Z)
    clf    = LogisticRegression(max_iter=500, C=1.0, solver='saga', n_jobs=-1)
    acc    = cross_val_score(clf, Z_s, labels, cv=5,
                             scoring='accuracy', n_jobs=-1)
    f1     = cross_val_score(clf, Z_s, labels, cv=5,
                             scoring='f1_macro', n_jobs=-1)
    return {
        'semantic_probe_acc': float(acc.mean()),
        'semantic_probe_std': float(acc.std()),
        'semantic_probe_f1':  float(f1.mean()),
    }


def probe_gap(token_acc: float, semantic_acc: float) -> float:
    """
    Probe Gap = semantic_acc − token_acc
    Positive gap → model encodes meaning over surface form.
    Larger gap → better representation quality for the task.
    """
    return semantic_acc - token_acc


# ── 3.4  Alignment & Uniformity (Wang & Isola, 2020) ─────────────────────────

def alignment_uniformity(Z: np.ndarray,
                         labels: np.ndarray,
                         alpha: float = 2.0,
                         t: float = 2.0) -> dict:
    """
    Alignment: average distance between embeddings of the same class.
        align(f; α) = E_{(x,y)~p_pos} [ ‖f(x) − f(y)‖² ]^α
    Lower is better — same-class reps cluster tightly.

    Uniformity: how uniformly the embeddings cover the hypersphere.
        uniform(f; t) = log E_{x,y~p_data} [ e^{−t‖f(x)−f(y)‖²} ]
    More negative is better — representations spread out maximally.

    Both computed on L2-normalised embeddings.
    """
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    Z_t = torch.from_numpy(Z_n).float()

    # ── Alignment ────────────────────────────────────────────────────────────
    align_vals = []
    unique_cls = np.unique(labels)
    for c in unique_cls:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        Zc = Z_t[idx]
        # All pairs within class
        diff   = Zc.unsqueeze(0) - Zc.unsqueeze(1)         # (n,n,D)
        sq_dist = (diff ** 2).sum(-1)                       # (n,n)
        # Upper triangle (no diagonal)
        mask   = torch.triu(torch.ones(len(idx), len(idx)), diagonal=1).bool()
        align_vals.append(sq_dist[mask].pow(alpha).mean().item())

    alignment = float(np.mean(align_vals)) if align_vals else float('nan')

    # ── Uniformity ───────────────────────────────────────────────────────────
    # Subsample for efficiency (≤1000)
    n = min(len(Z_t), 1000)
    idx_sub  = torch.randperm(len(Z_t))[:n]
    Z_sub    = Z_t[idx_sub]
    sq_dist  = torch.pdist(Z_sub).pow(2)
    uniformity = float(torch.log(torch.exp(-t * sq_dist).mean() + 1e-12).item())

    return {
        'alignment':   alignment,   # lower is better
        'uniformity':  uniformity,  # more negative is better
    }


def intra_inter_class_distances(Z: np.ndarray,
                                labels: np.ndarray) -> dict:
    """
    Ratio of mean intra-class distance to mean inter-class distance.
    Ratio < 1 means same-class points are closer than different-class points.
    Lower ratio → better class separation.
    """
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    intra, inter = [], []
    unique = np.unique(labels)
    for i, ci in enumerate(unique):
        idx_i = np.where(labels == ci)[0]
        if len(idx_i) < 2:
            continue
        Zi = Z_n[idx_i]
        d_intra = np.linalg.norm(
            Zi[:, None] - Zi[None, :], axis=-1
        )
        mask = np.triu(np.ones((len(Zi), len(Zi)), dtype=bool), k=1)
        intra.extend(d_intra[mask].tolist())

        for cj in unique[i+1:]:
            idx_j = np.where(labels == cj)[0]
            Zj    = Z_n[idx_j]
            d_inter = np.linalg.norm(
                Zi[:, None, :] - Zj[None, :, :], axis=-1
            ).ravel()
            inter.extend(d_inter.tolist())

    intra_mean = float(np.mean(intra)) if intra else float('nan')
    inter_mean = float(np.mean(inter)) if inter else float('nan')
    ratio      = intra_mean / (inter_mean + 1e-12)
    return {
        'intra_mean': intra_mean,
        'inter_mean': inter_mean,
        'intra_inter_ratio': ratio,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_eigenspectrum(results: dict, dataset_name: str, save_dir: str):
    """
    Two panels:
      Left  — top-50 eigenvalues (log scale)
      Right — cumulative explained variance
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Eigenvalue Spectrum  ·  {dataset_name.upper()}',
                 fontsize=13, fontweight='bold', y=1.01)

    top_k = 50
    for model_key, res in results.items():
        spec  = res['spectrum']
        color = COLORS[model_key]
        label = LABELS[model_key]

        # Left: eigenvalue decay
        eigvals = spec['eigenvalues'][:top_k]
        axes[0].plot(np.arange(1, len(eigvals)+1), eigvals,
                     color=color, linewidth=2, label=label, marker='o',
                     markersize=3)

        # Right: cumulative variance
        cum = spec['cumulative_var'][:top_k]
        axes[1].plot(np.arange(1, len(cum)+1), cum * 100,
                     color=color, linewidth=2, label=label)

    axes[0].set_xlabel('Component index')
    axes[0].set_ylabel('Eigenvalue (σ²)')
    axes[0].set_yscale('log')
    axes[0].set_title('Eigenvalue Decay  (log scale)\n'
                      'Flatter = richer representation', fontsize=10)
    axes[0].legend()

    axes[1].set_xlabel('Number of components')
    axes[1].set_ylabel('Cumulative explained variance (%)')
    axes[1].set_title('Cumulative Explained Variance\n'
                      'More components needed = higher effective dim', fontsize=10)
    axes[1].legend()
    axes[1].axhline(90, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[1].text(2, 91, '90% threshold', color='grey', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, f'eigenspectrum_{dataset_name}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_effective_rank(results_all: dict, save_dir: str):
    """Bar chart comparing effective-rank metrics across models and datasets."""
    metrics    = ['erank', 'rank_90', 'stable_rank']
    m_labels   = ['Effective Rank\n(Roy & Vetrov)', '90%-Var Rank', 'Stable Rank']
    datasets   = sorted({ds for (ds, _) in results_all.keys()})
    model_keys = sorted({mk for (_, mk) in results_all.keys()})

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    fig.suptitle('Effective Rank Comparison', fontsize=13, fontweight='bold')

    width  = 0.35
    x      = np.arange(len(datasets))

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        for i, mk in enumerate(model_keys):
            vals = [results_all.get((ds, mk), {}).get('erank_metrics', {}).get(metric, 0)
                    for ds in datasets]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=LABELS[mk],
                          color=COLORS[mk], alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.set_title(mlabel, fontsize=10)
        ax.set_ylabel('Rank')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'effective_rank.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_probe_comparison(results_all: dict, save_dir: str):
    """
    Grouped bars: token probe vs semantic probe accuracy for each model/dataset.
    Shows the probe gap visually.
    """
    datasets   = sorted({ds for (ds, _) in results_all.keys()})
    model_keys = sorted({mk for (_, mk) in results_all.keys()})

    n_ds  = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]
    fig.suptitle('Token Probe vs Semantic Probe Accuracy', fontsize=13,
                 fontweight='bold')

    for ax, ds in zip(axes, datasets):
        probe_types = ['Token Probe\n(surface form)', 'Semantic Probe\n(task meaning)']
        x = np.arange(len(probe_types))
        width = 0.3

        for i, mk in enumerate(model_keys):
            res = results_all.get((ds, mk), {})
            t_acc = res.get('token_probe', {}).get('token_probe_acc', 0) or 0
            s_acc = res.get('semantic_probe', {}).get('semantic_probe_acc', 0) or 0
            vals  = [t_acc * 100, s_acc * 100]
            t_std = (res.get('token_probe', {}).get('token_probe_std', 0) or 0) * 100
            s_std = (res.get('semantic_probe', {}).get('semantic_probe_std', 0) or 0) * 100
            errs  = [t_std, s_std]

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                          label=LABELS[mk], color=COLORS[mk],
                          alpha=0.85, edgecolor='white')

            # Annotate probe gap above semantic bar
            gap = s_acc - t_acc
            ax.text(x[1] + offset, vals[1] + errs[1] + 1.5,
                    f'Δ={gap*100:+.1f}%', ha='center', va='bottom',
                    fontsize=8, color=COLORS[mk], fontweight='bold')

        ax.set_title(f'{ds.upper()}\n(↑ semantic, ↓ token = better)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(probe_types)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'probe_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_alignment_uniformity(results_all: dict, save_dir: str):
    """
    Scatter plot of (uniformity, alignment) for each model/dataset.
    Ideal position: low alignment (tight clusters) + low uniformity (spread sphere).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Alignment & Uniformity (Wang & Isola, 2020)',
                 fontsize=13, fontweight='bold')

    datasets   = sorted({ds for (ds, _) in results_all.keys()})
    model_keys = sorted({mk for (_, mk) in results_all.keys()})

    # Left: Scatter (uniformity vs alignment)
    ax = axes[0]
    for mk in model_keys:
        for ds in datasets:
            res = results_all.get((ds, mk), {})
            au  = res.get('align_uniform', {})
            u   = au.get('uniformity',  None)
            a   = au.get('alignment',   None)
            if u is None or a is None:
                continue
            ax.scatter(u, a, s=120, color=COLORS[mk],
                       marker='o' if mk == 'text_jepa' else 's',
                       zorder=5, edgecolors='white', linewidths=1)
            ax.annotate(f'{LABELS[mk]}\n{ds.upper()}',
                        (u, a), textcoords='offset points',
                        xytext=(8, 4), fontsize=8, color=COLORS[mk])

    ax.set_xlabel('Uniformity  (↓ better — more spread)')
    ax.set_ylabel('Alignment   (↓ better — tighter clusters)')
    ax.set_title('Alignment vs Uniformity\n'
                 'Bottom-left corner = ideal', fontsize=10)

    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS['text_jepa'],
               markersize=9, label=LABELS['text_jepa']),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['bert'],
               markersize=9, label=LABELS['bert']),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    # Right: Intra/Inter class distance ratio
    ax2    = axes[1]
    x      = np.arange(len(datasets))
    width  = 0.3

    for i, mk in enumerate(model_keys):
        ratios = []
        for ds in datasets:
            res = results_all.get((ds, mk), {})
            r   = res.get('class_dist', {}).get('intra_inter_ratio', float('nan'))
            ratios.append(r)
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, ratios, width,
                       label=LABELS[mk], color=COLORS[mk],
                       alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, ratios):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.005,
                         f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax2.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax2.text(len(datasets)-0.5, 1.01, 'ratio = 1 (no separation)',
             color='grey', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.set_ylabel('Intra / Inter class distance (↓ better)')
    ax2.set_title('Class Separation\n'
                  'Ratio < 1 = same-class reps cluster together', fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'alignment_uniformity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_spectral_entropy(results_all: dict, save_dir: str):
    """
    Radar / spider chart of all 6 normalised metrics per model.
    Gives a holistic 'at-a-glance' view.
    """
    datasets   = sorted({ds for (ds, _) in results_all.keys()})
    model_keys = sorted({mk for (_, mk) in results_all.keys()})

    for ds in datasets:
        metric_names = [
            'Spectral\nEntropy',
            'Eff. Rank\n(norm)',
            'Semantic\nProbe',
            'Probe\nGap',
            'Uniformity\n(inv, norm)',
            'Class Sep.\n(inv, norm)',
        ]
        N = len(metric_names)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6),
                                subplot_kw=dict(polar=True))
        ax.set_title(f'Representation Quality Radar  ·  {ds.upper()}',
                     fontsize=11, fontweight='bold', pad=16)

        def normalise_across(key_fn):
            """Normalise a metric to [0,1] across both models."""
            vals = {mk: key_fn(results_all.get((ds, mk), {}))
                    for mk in model_keys}
            vs   = [v for v in vals.values() if v is not None and not np.isnan(v)]
            if not vs or max(vs) == min(vs):
                return {mk: 0.5 for mk in model_keys}
            lo, hi = min(vs), max(vs)
            return {mk: (vals[mk] - lo) / (hi - lo + 1e-12) for mk in model_keys}

        se_n   = normalise_across(lambda r: r.get('spectrum', {}).get('spectral_entropy', np.nan))
        er_n   = normalise_across(lambda r: r.get('erank_metrics', {}).get('erank', np.nan))
        sp_n   = normalise_across(lambda r: r.get('semantic_probe', {}).get('semantic_probe_acc', np.nan))
        pg_n   = normalise_across(lambda r: probe_gap(
                    r.get('token_probe', {}).get('token_probe_acc', 0) or 0,
                    r.get('semantic_probe', {}).get('semantic_probe_acc', 0) or 0))
        # Uniformity: more negative = better → invert
        uni_n_raw = normalise_across(lambda r: -(r.get('align_uniform', {}).get('uniformity', np.nan) or np.nan))
        # Class separation: lower ratio = better → invert
        cs_n_raw  = normalise_across(lambda r: 1 - (r.get('class_dist', {}).get('intra_inter_ratio', np.nan) or np.nan))
        uni_n = uni_n_raw
        cs_n  = cs_n_raw

        for mk in model_keys:
            values = [
                se_n.get(mk, 0),
                er_n.get(mk, 0),
                sp_n.get(mk, 0),
                pg_n.get(mk, 0),
                uni_n.get(mk, 0),
                cs_n.get(mk, 0),
            ]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, color=COLORS[mk],
                    label=LABELS[mk])
            ax.fill(angles, values, alpha=0.12, color=COLORS[mk])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25','0.5','0.75','1.0'], fontsize=7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax.grid(alpha=0.4)

        path = os.path.join(save_dir, f'radar_{ds}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_analysis(model, model_key: str, dataset_name: str,
                 tokenizer, args, device) -> dict:
    print(f'\n  [{model_key.upper()}] Loading {dataset_name} …')
    input_ids, attn_mask, labels_t, texts = load_split(
        dataset_name, args.split, tokenizer,
        args.max_length, args.max_samples, device
    )
    labels = labels_t.cpu().numpy()

    print(f'  [{model_key.upper()}] Extracting embeddings …')
    Z = extract_embeddings(model, input_ids, attn_mask,
                           batch_size=args.batch_size)

    print(f'  [{model_key.upper()}] Computing eigenspectrum …')
    spectrum = compute_eigenspectrum(Z)

    print(f'  [{model_key.upper()}] Computing effective rank …')
    erank_metrics = effective_rank(Z)

    print(f'  [{model_key.upper()}] Running token probe …')
    tok_probe = token_probe(Z, texts, tokenizer)

    print(f'  [{model_key.upper()}] Running semantic probe …')
    cfg       = DATASET_CFG[dataset_name]
    sem_probe = semantic_probe(Z, labels, cfg['label_names'])

    print(f'  [{model_key.upper()}] Computing alignment/uniformity …')
    au = alignment_uniformity(Z, labels)
    cd = intra_inter_class_distances(Z, labels)

    return {
        'spectrum':       spectrum,
        'erank_metrics':  erank_metrics,
        'token_probe':    tok_probe,
        'semantic_probe': sem_probe,
        'align_uniform':  au,
        'class_dist':     cd,
    }


def print_summary(results_all: dict):
    print('\n' + '═'*70)
    print('  REPRESENTATION ANALYSIS SUMMARY')
    print('═'*70)

    datasets   = sorted({ds for (ds, _) in results_all.keys()})
    model_keys = sorted({mk for (_, mk) in results_all.keys()})

    for ds in datasets:
        print(f'\n  Dataset: {ds.upper()}')
        print(f'  {"Metric":<35}', end='')
        for mk in model_keys:
            print(f'  {LABELS[mk]:<16}', end='')
        print()
        print('  ' + '─'*65)

        def row(label, key_fn):
            print(f'  {label:<35}', end='')
            for mk in model_keys:
                v = key_fn(results_all.get((ds, mk), {}))
                print(f'  {v:<16.4f}' if isinstance(v, float) else f'  {v!s:<16}', end='')
            print()

        row('Spectral Entropy (↑ richer)',
            lambda r: r.get('spectrum', {}).get('spectral_entropy', float('nan')))
        row('Effective Rank / erank (↑)',
            lambda r: r.get('erank_metrics', {}).get('erank', float('nan')))
        row('90%-Variance Rank (↑)',
            lambda r: r.get('erank_metrics', {}).get('rank_90', float('nan')))
        row('Stable Rank (↑)',
            lambda r: r.get('erank_metrics', {}).get('stable_rank', float('nan')))
        row('Token Probe Acc (↓ surface)',
            lambda r: r.get('token_probe', {}).get('token_probe_acc', float('nan')) or float('nan'))
        row('Semantic Probe Acc (↑)',
            lambda r: r.get('semantic_probe', {}).get('semantic_probe_acc', float('nan')))
        row('Probe Gap / Δ (↑)',
            lambda r: probe_gap(
                r.get('token_probe', {}).get('token_probe_acc', 0) or 0,
                r.get('semantic_probe', {}).get('semantic_probe_acc', 0) or 0))
        row('Alignment (↓ tighter clusters)',
            lambda r: r.get('align_uniform', {}).get('alignment', float('nan')))
        row('Uniformity (↓ more spread)',
            lambda r: r.get('align_uniform', {}).get('uniformity', float('nan')))
        row('Intra/Inter ratio (↓ separation)',
            lambda r: r.get('class_dist', {}).get('intra_inter_ratio', float('nan')))

    print('\n' + '═'*70 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Representation Analysis')
    parser.add_argument('--text_jepa_ckpt', type=str, required=True,
                        help='Path to hybrid (JEPA+MLM) checkpoint (.pth.tar)')
    parser.add_argument('--mlm_ckpt',       type=str, required=True,
                        help='Path to pure-MLM baseline checkpoint (.pth.tar)')
    parser.add_argument('--tokenizer',      type=str, default='bert-base-uncased',
                        help='HuggingFace tokenizer name (vocab only, no model loaded)')
    parser.add_argument('--pad_token_id',   type=int, default=0,
                        help='Padding token id (default: 0, matches your training config)')
    parser.add_argument('--datasets',       nargs='+',
                        default=['sst2', 'mrpc'],
                        choices=['sst2', 'mrpc'])
    parser.add_argument('--split',          type=str, default='validation')
    parser.add_argument('--max_samples',    type=int, default=2000)
    parser.add_argument('--max_length',     type=int, default=128)
    parser.add_argument('--batch_size',     type=int, default=64)
    parser.add_argument('--output_dir',     type=str, default='analysis_results')
    parser.add_argument('--device',         type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer — used for vocab / token ids only, not for model weights ────
    # Both checkpoints were trained with the same tokenizer, so one is enough.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ── Load both encoders from YOUR checkpoints ──────────────────────────────
    print('\nLoading Text-JEPA encoder (hybrid checkpoint) …')
    text_jepa = EncoderWrapper(args.text_jepa_ckpt, device,
                               pad_token_id=args.pad_token_id)

    print('\nLoading MLM-baseline encoder (mlm checkpoint) …')
    bert = EncoderWrapper(args.mlm_ckpt, device,
                          pad_token_id=args.pad_token_id)

    models = {'text_jepa': text_jepa, 'bert': bert}

    # ── Run all analyses ──────────────────────────────────────────────────────
    results_all = {}
    for ds in args.datasets:
        for mk, model in models.items():
            key = (ds, mk)
            results_all[key] = run_analysis(
                model, mk, ds, tokenizer, args, device
            )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating plots …')
    for ds in args.datasets:
        ds_results = {mk: results_all[(ds, mk)]
                      for mk in models if (ds, mk) in results_all}
        plot_eigenspectrum({'text_jepa': results_all.get(('sst2','text_jepa'), {}),
                             'bert':      results_all.get(('sst2','bert'), {})},
                            ds, args.output_dir)

    plot_effective_rank(results_all, args.output_dir)
    plot_probe_comparison(results_all, args.output_dir)
    plot_alignment_uniformity(results_all, args.output_dir)
    plot_spectral_entropy(results_all, args.output_dir)

    # ── Print and save summary ─────────────────────────────────────────────────
    print_summary(results_all)

    # Serialise numbers (not arrays) to JSON
    def serialise(obj):
        if isinstance(obj, (np.integer,)):       return int(obj)
        if isinstance(obj, (np.floating,)):      return float(obj)
        if isinstance(obj, np.ndarray):          return obj[:50].tolist()
        if isinstance(obj, dict):
            return {k: serialise(v) for k, v in obj.items()}
        return obj

    summary_path = os.path.join(args.output_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({str(k): serialise(v) for k, v in results_all.items()},
                  f, indent=2)
    print(f'  JSON summary saved: {summary_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()