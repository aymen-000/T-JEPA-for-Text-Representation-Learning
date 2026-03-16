"""
Representation Analysis  —  Hybrid vs MLM-only Encoder
=======================================================
Investigates encoder representation quality across 4 axes:

  1. Eigenvalue Spectrum       — how "spread" the representation space is
  2. Effective Rank            — dimensionality actually used by the encoder
  3. Token vs Semantic Probes  — does the encoder encode surface form or meaning?
  4. Alignment & Uniformity    — geometric quality of the embedding hypersphere

Pooling ablation (--pooling flag):
  mean      : mean over non-padding tokens  (default, matches fine-tuning)
  max       : element-wise max over non-padding tokens
  weighted  : learned scalar weights per position (softmax-normalised)
  attention : lightweight single-head attention pooling

Inspired by:
  - Wang & Isola (2020)   Alignment and Uniformity on the Hypersphere
  - Roy & Vetrov (2007)   The Effective Rank
  - Ethayarajh (2019)     How Contextual are Contextualized Word Representations?
  - Garrido et al. (2023) Duality Between Contrastive and Non-Contrastive SSL
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import entropy as scipy_entropy, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Visual style
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'hybrid': '#E63946',   # vivid red
    'mlm':    '#457B9D',   # steel blue
}
LABELS = {
    'hybrid': 'Hybrid (JEPA+MLM)',
    'mlm':    'MLM-only',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.6,
    'figure.dpi':        150,
})


# ═════════════════════════════════════════════════════════════════════════════
# 1.  POOLING STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

class MeanPooling(nn.Module):
    """Mean over non-padding tokens — matches TextLinearProbeModel exactly."""
    def forward(self, hidden: torch.Tensor,
                input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        mask = (input_ids != pad_id).unsqueeze(-1).float()   # (B, L, 1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class MaxPooling(nn.Module):
    """Element-wise max over non-padding tokens."""
    def forward(self, hidden: torch.Tensor,
                input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        mask = (input_ids != pad_id).unsqueeze(-1)           # (B, L, 1) bool
        # Replace padding positions with -inf before max
        hidden = hidden.masked_fill(~mask, float('-inf'))
        return hidden.max(dim=1).values                      # (B, D)


class WeightedMeanPooling(nn.Module):
    """
    Learned scalar weight per position, softmax-normalised over non-padding.
    Only the weight vector is trainable — the encoder stays frozen.
    """
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(max_seq_len))

    def forward(self, hidden: torch.Tensor,
                input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        B, L, D = hidden.shape
        w = self.weights[:L]                                  # (L,)
        pad_mask = (input_ids == pad_id)                      # (B, L) True = pad
        w_exp = w.unsqueeze(0).expand(B, -1)                  # (B, L)
        w_exp = w_exp.masked_fill(pad_mask, float('-inf'))
        w_norm = torch.softmax(w_exp, dim=1).unsqueeze(-1)    # (B, L, 1)
        return (hidden * w_norm).sum(1)                       # (B, D)


class AttentionPooling(nn.Module):
    """
    Lightweight single-head attention pooling.
    A learned query attends over all non-padding token representations.
    Architecture: query vector → dot-product scores → softmax → weighted sum.
    Only query + key projection are trainable.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.key.weight)

    def forward(self, hidden: torch.Tensor,
                input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        # hidden : (B, L, D)
        keys    = self.key(hidden)                            # (B, L, D)
        scores  = torch.einsum('d,bld->bl', self.query, keys) # (B, L)
        scores  = scores / (hidden.size(-1) ** 0.5)
        pad_mask = (input_ids == pad_id)
        scores   = scores.masked_fill(pad_mask, float('-inf'))
        weights  = torch.softmax(scores, dim=1).unsqueeze(-1) # (B, L, 1)
        return (hidden * weights).sum(1)                      # (B, D)


def build_pooler(pooling: str, embed_dim: int,
                 max_seq_len: int, device: torch.device) -> nn.Module:
    """Factory — returns the requested pooling module on the correct device."""
    if pooling == 'mean':
        return MeanPooling().to(device)
    if pooling == 'max':
        return MaxPooling().to(device)
    if pooling == 'weighted':
        return WeightedMeanPooling(max_seq_len).to(device)
    if pooling == 'attention':
        return AttentionPooling(embed_dim).to(device)
    raise ValueError(f"Unknown pooling: '{pooling}'. "
                     f"Choose from mean / max / weighted / attention.")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _load_encoder_from_ckpt(ckpt_path: str, device: torch.device) -> tuple:
    """
    Returns (encoder, embed_dim, max_seq_len).
    Reads all hyper-parameters from the checkpoint's embedded 'config' dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    required = {'encoder', 'config', 'training_mode'}
    if not required.issubset(ckpt.keys()):
        raise ValueError(
            f"Checkpoint '{ckpt_path}' missing keys. "
            f"Expected {required}, got {set(ckpt.keys())}."
        )

    cfg  = ckpt['config']
    mode = ckpt.get('training_mode', 'unknown')
    print(f"    training_mode = '{mode}'")

    from src.help.schedulers import init_model
    encoder, _ = init_model(
        device       = device,
        model_name   = cfg['meta']['model_name'],
        pred_depth   = cfg['meta']['pred_depth'],
        pred_emb_dim = cfg['meta']['pred_emb_dim'],
        vocab_size   = cfg['data'].get('vocab_size',  30522),
        max_seq_len  = cfg['data'].get('max_seq_len', 512),
    )
    encoder.load_state_dict(ckpt['encoder'])
    encoder.to(device).eval()

    embed_dim   = encoder.token_embed.token_embed.weight.shape[1]
    max_seq_len = cfg['data'].get('max_seq_len', 512)
    return encoder, embed_dim, max_seq_len


class EncoderWrapper(nn.Module):
    """
    Wraps a frozen encoder + a pluggable pooling strategy.
    Pooling modules with learnable parameters (weighted, attention) are
    also frozen after a brief warm-up fit — for the analysis we want to
    measure the encoder geometry, not train new parameters.
    """
    def __init__(self, ckpt_path: str, device: torch.device,
                 pooling: str = 'mean', pad_token_id: int = 0):
        super().__init__()
        print(f"  Loading: {ckpt_path}  |  pooling={pooling}")
        self.encoder, self.embed_dim, self.max_seq_len = \
            _load_encoder_from_ckpt(ckpt_path, device)
        self.pooler       = build_pooler(pooling, self.embed_dim,
                                         self.max_seq_len, device)
        self.pad_token_id = pad_token_id
        self.pooling_name = pooling
        self.device       = device

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encoder(input_ids)              # (B, L, D)
        # WeightedMeanPooling / AttentionPooling have learnable params
        # but we call them inside no_grad — they are fixed at init values
        # which is intentional: we measure the raw encoder geometry.
        return self.pooler(hidden, input_ids, self.pad_token_id)  # (B, D)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  DATASET HELPERS
# ═════════════════════════════════════════════════════════════════════════════

DATASET_CFG = {
    'sst2': {
        'hf_path':     ('glue', 'sst2'),
        'text_col':    'sentence',
        'label_col':   'label',
        'task':        'classification',
        'n_classes':   2,
        'label_names': ['negative', 'positive'],
        'split':       'validation',
        'metric':      'accuracy',
    },
    'mrpc': {
        'hf_path':     ('glue', 'mrpc'),
        'text_col':    ('sentence1', 'sentence2'),
        'label_col':   'label',
        'task':        'classification',
        'n_classes':   2,
        'label_names': ['not-paraphrase', 'paraphrase'],
        'split':       'validation',
        'metric':      'f1',
    },
    'mnli': {
        'hf_path':     ('glue', 'mnli'),
        'text_col':    ('premise', 'hypothesis'),
        'label_col':   'label',
        'task':        'classification',
        'n_classes':   3,
        'label_names': ['entailment', 'neutral', 'contradiction'],
        'split':       'validation_matched',
        'metric':      'accuracy',
    },
    'cola': {
        'hf_path':     ('glue', 'cola'),
        'text_col':    'sentence',
        'label_col':   'label',
        'task':        'classification',
        'n_classes':   2,
        'label_names': ['unacceptable', 'acceptable'],
        'split':       'validation',
        'metric':      'mcc',
    },
    'stsb': {
        'hf_path':     ('glue', 'stsb'),
        'text_col':    ('sentence1', 'sentence2'),
        'label_col':   'label',
        'task':        'regression',
        'n_classes':   None,
        'label_names': None,
        'split':       'validation',
        'metric':      'spearman',
    },
}


def load_split(dataset_name: str, split: str, tokenizer,
               max_length: int, max_samples: int, device: torch.device):
    """
    Returns (input_ids, attention_mask, labels, raw_texts).
    For STS-B labels are float; for others int.
    """
    cfg  = DATASET_CFG[dataset_name]
    ds   = load_dataset(*cfg['hf_path'], split=split or cfg['split'])

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    tc = cfg['text_col']
    if isinstance(tc, tuple):
        texts = [f"{row[tc[0]]} [SEP] {row[tc[1]]}" for row in ds]
    else:
        texts = [row[tc] for row in ds]

    raw_labels = [row[cfg['label_col']] for row in ds]

    enc = tokenizer(texts, max_length=max_length, padding='max_length',
                    truncation=True, return_tensors='pt')

    if cfg['task'] == 'regression':
        labels_t = torch.tensor(raw_labels, dtype=torch.float)
    else:
        labels_t = torch.tensor(raw_labels, dtype=torch.long)

    return (enc['input_ids'].to(device),
            enc['attention_mask'].to(device),
            labels_t.to(device),
            texts)


@torch.no_grad()
def extract_embeddings(model: EncoderWrapper, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, input_ids.size(0), batch_size):
        emb = model.encode(input_ids[i:i+batch_size],
                           attention_mask[i:i+batch_size])
        all_embs.append(emb.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ANALYSIS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 4.1  Eigenvalue Spectrum ──────────────────────────────────────────────────

def compute_eigenspectrum(Z: np.ndarray) -> dict:
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
    eigvals = S ** 2
    total   = eigvals.sum()
    exp_var = eigvals / (total + 1e-12)
    cum_var = np.cumsum(exp_var)
    p       = exp_var / (exp_var.sum() + 1e-12)
    spec_entropy = float(scipy_entropy(p) / np.log(len(p) + 1e-12))
    return {'eigenvalues': eigvals, 'explained_var': exp_var,
            'cumulative_var': cum_var, 'spectral_entropy': spec_entropy}


# ── 4.2  Effective Rank ───────────────────────────────────────────────────────

def effective_rank(Z: np.ndarray) -> dict:
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
    p     = S / (S.sum() + 1e-12)
    p     = p[p > 1e-12]
    erank = float(np.exp(-np.sum(p * np.log(p))))
    eigvals = S ** 2
    cum_var = np.cumsum(eigvals) / (eigvals.sum() + 1e-12)
    rank_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    stable_rank = float((S ** 2).sum() / (S[0] ** 2 + 1e-12))
    return {'erank': erank, 'rank_90': rank_90,
            'stable_rank': stable_rank, 'singular_vals': S}


# ── 4.3  Probes ───────────────────────────────────────────────────────────────

def token_probe(Z: np.ndarray, texts: list, tokenizer) -> dict:
    """Surface-form probe: predict most-frequent token from embedding."""
    special_ids = set(tokenizer.all_special_ids)
    targets = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        ids = [i for i in ids if i not in special_ids]
        targets.append(int(np.bincount(ids).argmax()) if ids else 0)

    targets = np.array(targets)
    classes, counts = np.unique(targets, return_counts=True)
    valid   = classes[counts >= 5]
    mask    = np.isin(targets, valid)
    Z_sub, y_sub = Z[mask], targets[mask]

    if len(np.unique(y_sub)) < 2:
        return {'token_probe_acc': float('nan'), 'token_probe_std': float('nan')}

    Z_s    = StandardScaler().fit_transform(Z_sub)
    clf    = LogisticRegression(max_iter=500, C=1.0, solver='saga',
                                multi_class='auto', n_jobs=-1)
    scores = cross_val_score(clf, Z_s, y_sub, cv=3, scoring='accuracy')
    return {'token_probe_acc': float(scores.mean()),
            'token_probe_std': float(scores.std())}


def semantic_probe(Z: np.ndarray, labels: np.ndarray,
                   task: str, label_names: list | None) -> dict:
    """
    Semantic probe adapted per task:
      classification → logistic regression accuracy / f1 / MCC
      regression     → ridge regression Spearman ρ
    """
    Z_s = StandardScaler().fit_transform(Z)

    if task == 'regression':
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict
        preds = cross_val_predict(Ridge(), Z_s, labels, cv=5)
        r, _  = spearmanr(labels, preds)
        return {'semantic_probe_acc': float(r),
                'semantic_probe_std': float('nan'),
                'semantic_probe_f1':  float('nan'),
                'semantic_probe_mcc': float('nan')}

    clf = LogisticRegression(max_iter=500, C=1.0, solver='saga', n_jobs=-1)
    acc = cross_val_score(clf, Z_s, labels, cv=5, scoring='accuracy')
    f1  = cross_val_score(clf, Z_s, labels, cv=5, scoring='f1_macro')

    # MCC via cross_val_predict for CoLA
    if task == 'classification' and len(np.unique(labels)) == 2:
        from sklearn.model_selection import cross_val_predict
        preds = cross_val_predict(clf, Z_s, labels, cv=5)
        mcc   = float(matthews_corrcoef(labels, preds))
    else:
        mcc = float('nan')

    return {'semantic_probe_acc': float(acc.mean()),
            'semantic_probe_std': float(acc.std()),
            'semantic_probe_f1':  float(f1.mean()),
            'semantic_probe_mcc': mcc}


def probe_gap(token_acc: float, semantic_acc: float) -> float:
    return semantic_acc - token_acc


# ── 4.4  Alignment & Uniformity ──────────────────────────────────────────────

def alignment_uniformity(Z: np.ndarray, labels: np.ndarray,
                         alpha: float = 2.0, t: float = 2.0) -> dict:
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    Z_t = torch.from_numpy(Z_n).float()

    align_vals = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2: continue
        Zc      = Z_t[idx]
        diff    = Zc.unsqueeze(0) - Zc.unsqueeze(1)
        sq_dist = (diff ** 2).sum(-1)
        mask    = torch.triu(torch.ones(len(idx), len(idx)), diagonal=1).bool()
        align_vals.append(sq_dist[mask].pow(alpha).mean().item())

    alignment = float(np.mean(align_vals)) if align_vals else float('nan')

    n        = min(len(Z_t), 1000)
    Z_sub    = Z_t[torch.randperm(len(Z_t))[:n]]
    sq_dist  = torch.pdist(Z_sub).pow(2)
    uniformity = float(torch.log(torch.exp(-t * sq_dist).mean() + 1e-12).item())

    return {'alignment': alignment, 'uniformity': uniformity}


def intra_inter_class_distances(Z: np.ndarray, labels: np.ndarray) -> dict:
    Z_n   = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    intra, inter = [], []
    unique = np.unique(labels)
    for i, ci in enumerate(unique):
        idx_i = np.where(labels == ci)[0]
        if len(idx_i) < 2: continue
        Zi = Z_n[idx_i]
        d  = np.linalg.norm(Zi[:, None] - Zi[None, :], axis=-1)
        m  = np.triu(np.ones((len(Zi), len(Zi)), dtype=bool), k=1)
        intra.extend(d[m].tolist())
        for cj in unique[i+1:]:
            Zj = Z_n[np.where(labels == cj)[0]]
            inter.extend(np.linalg.norm(
                Zi[:, None, :] - Zj[None, :, :], axis=-1).ravel().tolist())

    intra_mean = float(np.mean(intra)) if intra else float('nan')
    inter_mean = float(np.mean(inter)) if inter else float('nan')
    return {'intra_mean': intra_mean, 'inter_mean': inter_mean,
            'intra_inter_ratio': intra_mean / (inter_mean + 1e-12)}


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def _model_keys(results_all): return sorted({mk for (_, mk, _) in results_all})
def _pool_keys(results_all):  return sorted({pl for (_, _, pl) in results_all})
def _ds_keys(results_all):    return sorted({ds for (ds, _, _) in results_all})


def plot_eigenspectrum(results_all: dict, save_dir: str):
    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)
    top_k      = 50

    for ds in datasets:
        n_pools = len(pool_keys)
        fig, axes = plt.subplots(2, n_pools, figsize=(5 * n_pools, 8))
        if n_pools == 1: axes = axes.reshape(2, 1)
        fig.suptitle(f'Eigenvalue Spectrum  ·  {ds.upper()}',
                     fontsize=13, fontweight='bold')

        for col, pool in enumerate(pool_keys):
            for mk in model_keys:
                res  = results_all.get((ds, mk, pool), {})
                spec = res.get('spectrum', {})
                if not spec: continue
                color = COLORS[mk]
                label = LABELS[mk]

                eigvals = spec['eigenvalues'][:top_k]
                axes[0, col].plot(np.arange(1, len(eigvals)+1), eigvals,
                                  color=color, linewidth=2, label=label,
                                  marker='o', markersize=3)

                cum = spec['cumulative_var'][:top_k]
                axes[1, col].plot(np.arange(1, len(cum)+1), cum * 100,
                                  color=color, linewidth=2, label=label)

            axes[0, col].set_title(f'Pooling: {pool}\nEigenvalue Decay (log)', fontsize=9)
            axes[0, col].set_yscale('log')
            axes[0, col].set_xlabel('Component')
            axes[0, col].set_ylabel('Eigenvalue (σ²)')
            axes[0, col].legend(fontsize=8)

            axes[1, col].set_title('Cumulative Explained Variance', fontsize=9)
            axes[1, col].set_xlabel('Components')
            axes[1, col].set_ylabel('Cum. Var (%)')
            axes[1, col].axhline(90, color='grey', linestyle='--',
                                 linewidth=0.8, alpha=0.7)
            axes[1, col].legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(save_dir, f'eigenspectrum_{ds}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_effective_rank(results_all: dict, save_dir: str):
    metrics    = ['erank', 'rank_90', 'stable_rank']
    m_labels   = ['Effective Rank\n(Roy & Vetrov)', '90%-Var Rank', 'Stable Rank']
    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)

    for pool in pool_keys:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(f'Effective Rank  ·  pooling={pool}',
                     fontsize=13, fontweight='bold')
        width = 0.35
        x     = np.arange(len(datasets))

        for ax, metric, mlabel in zip(axes, metrics, m_labels):
            for i, mk in enumerate(model_keys):
                vals   = [results_all.get((ds, mk, pool), {})
                          .get('erank_metrics', {}).get(metric, 0)
                          for ds in datasets]
                offset = (i - 0.5) * width
                bars   = ax.bar(x + offset, vals, width, label=LABELS[mk],
                                color=COLORS[mk], alpha=0.85, edgecolor='white')
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.3, f'{v:.1f}',
                            ha='center', va='bottom', fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([d.upper() for d in datasets])
            ax.set_title(mlabel, fontsize=10)
            ax.set_ylabel('Rank')
            ax.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(save_dir, f'effective_rank_pool_{pool}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_probe_comparison(results_all: dict, save_dir: str):
    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)

    for pool in pool_keys:
        n_ds  = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
        if n_ds == 1: axes = [axes]
        fig.suptitle(f'Token vs Semantic Probe  ·  pooling={pool}',
                     fontsize=13, fontweight='bold')

        for ax, ds in zip(axes, datasets):
            x     = np.arange(2)
            width = 0.3
            for i, mk in enumerate(model_keys):
                res   = results_all.get((ds, mk, pool), {})
                t_acc = res.get('token_probe', {}).get('token_probe_acc', 0) or 0
                s_acc = res.get('semantic_probe', {}).get('semantic_probe_acc', 0) or 0
                t_std = (res.get('token_probe', {}).get('token_probe_std', 0) or 0)
                s_std = (res.get('semantic_probe', {}).get('semantic_probe_std', 0) or 0)
                vals  = [t_acc * 100, s_acc * 100]
                errs  = [t_std * 100, s_std * 100]
                offset = (i - 0.5) * width
                ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                       label=LABELS[mk], color=COLORS[mk],
                       alpha=0.85, edgecolor='white')
                gap = s_acc - t_acc
                ax.text(x[1] + offset, vals[1] + errs[1] + 1.5,
                        f'Δ={gap*100:+.1f}%', ha='center', va='bottom',
                        fontsize=8, color=COLORS[mk], fontweight='bold')

            ax.set_title(f'{ds.upper()}', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(['Token Probe\n(surface)', 'Semantic Probe\n(task)'])
            ax.set_ylabel('Accuracy / Corr (%)')
            ax.set_ylim(0, 110)
            ax.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(save_dir, f'probe_comparison_pool_{pool}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_alignment_uniformity(results_all: dict, save_dir: str):
    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)

    # One figure per pooling strategy
    for pool in pool_keys:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'Alignment & Uniformity  ·  pooling={pool}',
                     fontsize=13, fontweight='bold')

        ax = axes[0]
        for mk in model_keys:
            for ds in datasets:
                res = results_all.get((ds, mk, pool), {})
                au  = res.get('align_uniform', {})
                u, a = au.get('uniformity'), au.get('alignment')
                if u is None or a is None: continue
                ax.scatter(u, a, s=120, color=COLORS[mk],
                           marker='o' if mk == 'hybrid' else 's',
                           zorder=5, edgecolors='white', linewidths=1)
                ax.annotate(f'{LABELS[mk]}\n{ds.upper()}', (u, a),
                            textcoords='offset points', xytext=(8, 4),
                            fontsize=7, color=COLORS[mk])

        ax.set_xlabel('Uniformity (↓ better)')
        ax.set_ylabel('Alignment  (↓ better)')
        ax.set_title('Alignment vs Uniformity\nbottom-left = ideal', fontsize=10)
        ax.legend(handles=[
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=COLORS['hybrid'], markersize=9,
                   label=LABELS['hybrid']),
            Line2D([0],[0], marker='s', color='w',
                   markerfacecolor=COLORS['mlm'],    markersize=9,
                   label=LABELS['mlm']),
        ], fontsize=9)

        ax2    = axes[1]
        x      = np.arange(len(datasets))
        width  = 0.3
        for i, mk in enumerate(model_keys):
            ratios = [results_all.get((ds, mk, pool), {})
                      .get('class_dist', {}).get('intra_inter_ratio', float('nan'))
                      for ds in datasets]
            offset = (i - 0.5) * width
            bars   = ax2.bar(x + offset, ratios, width,
                             label=LABELS[mk], color=COLORS[mk],
                             alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, ratios):
                if not np.isnan(v):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.003,
                             f'{v:.3f}', ha='center', va='bottom', fontsize=7)

        ax2.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.upper() for d in datasets])
        ax2.set_ylabel('Intra / Inter ratio (↓ better)')
        ax2.set_title('Class Separation', fontsize=10)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(save_dir, f'alignment_uniformity_pool_{pool}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_pooling_ablation(results_all: dict, save_dir: str):
    """
    Ablation: side-by-side bar chart comparing all pooling strategies
    for a fixed model on each dataset.
    Metric shown: spectral entropy + semantic probe accuracy.
    """
    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)

    if len(pool_keys) < 2:
        return   # ablation only meaningful with multiple poolings

    metrics = [
        ('spectral_entropy',    lambda r: r.get('spectrum', {}).get('spectral_entropy', np.nan)),
        ('semantic_probe_acc',  lambda r: r.get('semantic_probe', {}).get('semantic_probe_acc', np.nan)),
        ('uniformity',          lambda r: r.get('align_uniform', {}).get('uniformity', np.nan)),
    ]
    m_titles = ['Spectral Entropy (↑)', 'Semantic Probe Acc (↑)', 'Uniformity (↓)']

    for mk in model_keys:
        for ds in datasets:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig.suptitle(f'Pooling Ablation  ·  {LABELS[mk]}  ·  {ds.upper()}',
                         fontsize=12, fontweight='bold')

            x = np.arange(len(pool_keys))
            for ax, (_, key_fn), title in zip(axes, metrics, m_titles):
                vals   = [key_fn(results_all.get((ds, mk, pool), {}))
                          for pool in pool_keys]
                colors = plt.cm.Set2(np.linspace(0, 1, len(pool_keys)))
                bars   = ax.bar(x, vals, color=colors, edgecolor='white',
                                alpha=0.9)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + abs(bar.get_height()) * 0.02,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(pool_keys)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Pooling strategy')

            plt.tight_layout()
            path = os.path.join(save_dir,
                                f'pooling_ablation_{mk}_{ds}.png')
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {path}')


def plot_radar(results_all: dict, save_dir: str):
    """Per-dataset radar chart — one subplot per pooling strategy."""
    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)

    metric_names = ['Spectral\nEntropy', 'Eff. Rank', 'Semantic\nProbe',
                    'Probe Gap', 'Uniformity\n(inv)', 'Class Sep.\n(inv)']
    N      = len(metric_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + \
             [0 / float(N) * 2 * np.pi]

    def norm_across(key_fn, ds, pool):
        vals = {mk: key_fn(results_all.get((ds, mk, pool), {}))
                for mk in model_keys}
        vs   = [v for v in vals.values()
                if v is not None and not np.isnan(v)]
        if not vs or max(vs) == min(vs):
            return {mk: 0.5 for mk in model_keys}
        lo, hi = min(vs), max(vs)
        return {mk: (vals[mk] - lo) / (hi - lo + 1e-12) for mk in model_keys}

    for ds in datasets:
        n_pools = len(pool_keys)
        fig, axs = plt.subplots(1, n_pools, figsize=(6 * n_pools, 6),
                                subplot_kw=dict(polar=True))
        if n_pools == 1: axs = [axs]
        fig.suptitle(f'Representation Quality Radar  ·  {ds.upper()}',
                     fontsize=12, fontweight='bold')

        for ax, pool in zip(axs, pool_keys):
            se  = norm_across(lambda r: r.get('spectrum',{}).get('spectral_entropy',np.nan), ds, pool)
            er  = norm_across(lambda r: r.get('erank_metrics',{}).get('erank',np.nan), ds, pool)
            sp  = norm_across(lambda r: r.get('semantic_probe',{}).get('semantic_probe_acc',np.nan), ds, pool)
            pg  = norm_across(lambda r: probe_gap(
                    r.get('token_probe',{}).get('token_probe_acc',0) or 0,
                    r.get('semantic_probe',{}).get('semantic_probe_acc',0) or 0), ds, pool)
            uni = norm_across(lambda r: -(r.get('align_uniform',{}).get('uniformity',np.nan) or np.nan), ds, pool)
            cs  = norm_across(lambda r: 1-(r.get('class_dist',{}).get('intra_inter_ratio',np.nan) or np.nan), ds, pool)

            for mk in model_keys:
                vals   = [se[mk], er[mk], sp[mk], pg[mk], uni[mk], cs[mk]]
                vals  += vals[:1]
                ax.plot(angles, vals, linewidth=2, color=COLORS[mk], label=LABELS[mk])
                ax.fill(angles, vals, alpha=0.12, color=COLORS[mk])

            ax.set_title(f'pooling={pool}', fontsize=9, pad=14)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names, fontsize=8)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)

        plt.tight_layout()
        path = os.path.join(save_dir, f'radar_{ds}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 6.  RUN + SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def run_analysis(model: EncoderWrapper, model_key: str,
                 dataset_name: str, pooling: str,
                 tokenizer, args, device) -> dict:

    cfg   = DATASET_CFG[dataset_name]
    split = args.split or cfg['split']

    print(f'\n  [{model_key.upper()} | pool={pooling} | {dataset_name}] loading data …')
    input_ids, attn_mask, labels_t, texts = load_split(
        dataset_name, split, tokenizer,
        args.max_length, args.max_samples, device
    )

    if cfg['task'] == 'regression':
        labels = labels_t.cpu().numpy().astype(float)
        # For alignment/uniformity we need discrete labels — bin into 5 groups
        labels_disc = np.digitize(labels, bins=[1, 2, 3, 4]).astype(int)
    else:
        labels      = labels_t.cpu().numpy().astype(int)
        labels_disc = labels

    print(f'  Extracting embeddings …')
    Z = extract_embeddings(model, input_ids, attn_mask, args.batch_size)

    print(f'  Computing eigenspectrum …')
    spectrum = compute_eigenspectrum(Z)

    print(f'  Computing effective rank …')
    erank_m = effective_rank(Z)

    print(f'  Token probe …')
    tok = token_probe(Z, texts, tokenizer)

    print(f'  Semantic probe …')
    sem = semantic_probe(Z, labels, cfg['task'], cfg['label_names'])

    print(f'  Alignment / uniformity …')
    au = alignment_uniformity(Z, labels_disc)
    cd = intra_inter_class_distances(Z, labels_disc)

    return {'spectrum': spectrum, 'erank_metrics': erank_m,
            'token_probe': tok, 'semantic_probe': sem,
            'align_uniform': au, 'class_dist': cd}


def print_summary(results_all: dict):
    print('\n' + '═'*80)
    print('  REPRESENTATION ANALYSIS SUMMARY')
    print('═'*80)

    datasets   = _ds_keys(results_all)
    model_keys = _model_keys(results_all)
    pool_keys  = _pool_keys(results_all)

    for ds in datasets:
        for pool in pool_keys:
            print(f'\n  Dataset: {ds.upper()}   Pooling: {pool}')
            print(f'  {"Metric":<38}', end='')
            for mk in model_keys:
                print(f'  {LABELS[mk]:<20}', end='')
            print()
            print('  ' + '─'*78)

            def row(label, key_fn):
                print(f'  {label:<38}', end='')
                for mk in model_keys:
                    v = key_fn(results_all.get((ds, mk, pool), {}))
                    if isinstance(v, float):
                        print(f'  {v:<20.4f}', end='')
                    else:
                        print(f'  {str(v):<20}', end='')
                print()

            row('Spectral Entropy (↑)',
                lambda r: r.get('spectrum',{}).get('spectral_entropy', float('nan')))
            row('Effective Rank / erank (↑)',
                lambda r: r.get('erank_metrics',{}).get('erank', float('nan')))
            row('90%-Variance Rank (↑)',
                lambda r: r.get('erank_metrics',{}).get('rank_90', float('nan')))
            row('Stable Rank (↑)',
                lambda r: r.get('erank_metrics',{}).get('stable_rank', float('nan')))
            row('Token Probe Acc (↓)',
                lambda r: r.get('token_probe',{}).get('token_probe_acc', float('nan')) or float('nan'))
            row('Semantic Probe / Corr (↑)',
                lambda r: r.get('semantic_probe',{}).get('semantic_probe_acc', float('nan')))
            row('Semantic Probe F1 (↑)',
                lambda r: r.get('semantic_probe',{}).get('semantic_probe_f1', float('nan')))
            row('Semantic Probe MCC (↑, CoLA)',
                lambda r: r.get('semantic_probe',{}).get('semantic_probe_mcc', float('nan')))
            row('Probe Gap Δ (↑)',
                lambda r: probe_gap(
                    r.get('token_probe',{}).get('token_probe_acc', 0) or 0,
                    r.get('semantic_probe',{}).get('semantic_probe_acc', 0) or 0))
            row('Alignment (↓)',
                lambda r: r.get('align_uniform',{}).get('alignment', float('nan')))
            row('Uniformity (↓)',
                lambda r: r.get('align_uniform',{}).get('uniformity', float('nan')))
            row('Intra/Inter ratio (↓)',
                lambda r: r.get('class_dist',{}).get('intra_inter_ratio', float('nan')))

    print('\n' + '═'*80 + '\n')


# ═════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Representation Analysis — Hybrid vs MLM encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    parser.add_argument('--hybrid_ckpt',  type=str, required=True,
                        help='Hybrid (JEPA+MLM) checkpoint path')
    parser.add_argument('--mlm_ckpt',     type=str, required=True,
                        help='MLM-only baseline checkpoint path')

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    parser.add_argument('--tokenizer',    type=str, default='bert-base-uncased')
    parser.add_argument('--pad_token_id', type=int, default=0)

    # ── Datasets ──────────────────────────────────────────────────────────────
    parser.add_argument('--datasets', nargs='+',
                        default=['sst2', 'mrpc'],
                        choices=list(DATASET_CFG.keys()),
                        help='One or more GLUE datasets to evaluate on')
    parser.add_argument('--split',    type=str, default=None,
                        help='Dataset split override (default: per-dataset standard split)')

    # ── Pooling ───────────────────────────────────────────────────────────────
    parser.add_argument('--pooling', nargs='+',
                        default=['mean'],
                        choices=['mean', 'max', 'weighted', 'attention'],
                        help='One or more pooling strategies to ablate')

    # ── Sampling ──────────────────────────────────────────────────────────────
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--max_length',  type=int, default=128)
    parser.add_argument('--batch_size',  type=int, default=64)

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument('--output_dir',  type=str, default='analysis_results')
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--no_plots',    action='store_true',
                        help='Skip plot generation (faster, JSON only)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Models : hybrid={args.hybrid_ckpt}')
    print(f'         mlm   ={args.mlm_ckpt}')
    print(f'Datasets : {args.datasets}')
    print(f'Pooling  : {args.pooling}')

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ── Load one EncoderWrapper per (model_key, pooling) combo ────────────────
    model_ckpts = {'hybrid': args.hybrid_ckpt, 'mlm': args.mlm_ckpt}
    wrappers    = {}   # (model_key, pooling) → EncoderWrapper

    for mk, ckpt in model_ckpts.items():
        for pool in args.pooling:
            print(f'\nBuilding wrapper: model={mk}  pooling={pool}')
            wrappers[(mk, pool)] = EncoderWrapper(
                ckpt, device, pooling=pool, pad_token_id=args.pad_token_id
            )

    # ── Run analyses ──────────────────────────────────────────────────────────
    results_all = {}   # key: (dataset, model_key, pooling)

    for ds in args.datasets:
        for (mk, pool), model in wrappers.items():
            key = (ds, mk, pool)
            results_all[key] = run_analysis(
                model, mk, ds, pool, tokenizer, args, device
            )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print('\nGenerating plots …')
        plot_eigenspectrum(results_all, args.output_dir)
        plot_effective_rank(results_all, args.output_dir)
        plot_probe_comparison(results_all, args.output_dir)
        plot_alignment_uniformity(results_all, args.output_dir)
        plot_pooling_ablation(results_all, args.output_dir)
        plot_radar(results_all, args.output_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(results_all)

    # ── JSON dump ─────────────────────────────────────────────────────────────
    def serialise(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj[:50].tolist()
        if isinstance(obj, dict):        return {k: serialise(v) for k, v in obj.items()}
        return obj

    out_path = os.path.join(args.output_dir, 'results_summary.json')
    with open(out_path, 'w') as f:
        json.dump({str(k): serialise(v) for k, v in results_all.items()},
                  f, indent=2)
    print(f'\n  JSON saved: {out_path}')
    print('Done.')


if __name__ == '__main__':
    main()