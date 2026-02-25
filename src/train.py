import copy
import logging
import sys
import yaml
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.masks.all_masks import TextMutiBlockMaskCollector
from src.help.utils import apply_masks, repeat_interleave_batch, tokenize
from src.help.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter
)
from src.dataset.data.text_data import make_textjepa
from src.help.schedulers import (
    load_checkpoint,
    init_model,
    init_opt
)

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ---------------------------------------------------------
# MLM Head (BERT-style)
# ---------------------------------------------------------
class MLMHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.dense   = nn.Linear(embed_dim, embed_dim)
        self.norm    = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bias    = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense(hidden))
        x = self.norm(x)
        return self.decoder(x)


# ---------------------------------------------------------
# BERT-style MLM masking
# ---------------------------------------------------------
def apply_bert_masking(
    tokens: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    mlm_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_tokens = tokens.clone()
    mlm_labels    = torch.full_like(tokens, fill_value=-100)

    probability_matrix = torch.full(tokens.shape, mlm_prob, device=tokens.device)
    probability_matrix[tokens == pad_token_id] = 0.0

    masked_positions = torch.bernoulli(probability_matrix).bool()
    mlm_labels[masked_positions] = tokens[masked_positions]

    replace_with_mask = torch.bernoulli(
        torch.full(tokens.shape, 0.80, device=tokens.device)
    ).bool() & masked_positions
    masked_tokens[replace_with_mask] = mask_token_id

    replace_with_random = torch.bernoulli(
        torch.full(tokens.shape, 0.50, device=tokens.device)
    ).bool() & masked_positions & ~replace_with_mask
    random_tokens = torch.randint(
        low=0, high=vocab_size, size=tokens.shape,
        dtype=tokens.dtype, device=tokens.device
    )
    masked_tokens[replace_with_random] = random_tokens[replace_with_random]

    return masked_tokens, mlm_labels


# ---------------------------------------------------------
def main(args, resume_preempt=False):

    # ---------------- META ----------------
    use_bfloat16  = args['meta']['use_bfloat16']
    model_name    = args['meta']['model_name']
    load_model    = args['meta']['load_checkpoint'] or resume_preempt
    r_file        = args['meta']['read_checkpoint']
    pred_depth    = args['meta']['pred_depth']
    pred_emb_dim  = args['meta']['pred_emb_dim']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- DATA ----------------
    batch_size    = args['data']['batch_size']
    num_workers   = args['data']['num_workers']
    vocab_size    = args['data'].get('vocab_size', 30522)
    max_seq_len   = args['data'].get('max_seq_len', 512)
    mask_token_id = args['data'].get('mask_token_id', 103)
    pad_token_id  = args['data'].get('pad_token_id',  0)
    mlm_prob      = args['data'].get('mlm_prob', 0.15)

    # ---------------- MASK ----------------
    num_enc_masks   = args['mask']['num_enc_masks']
    num_pred_masks  = args['mask']['num_pred_masks']
    enc_mask_scale  = args['mask']['enc_mask_scale']
    pred_mask_scale = args['mask']['pred_mask_scale']
    min_keep        = args['mask']['min_keep']
    allow_overlap   = args['mask']['allow_overlap']
    max_tokens      = args['mask'].get('max_tokens', max_seq_len)

    # ---------------- OPT ----------------
    ema        = args['optimization']['ema']
    ipe_scale  = args['optimization'].get('ipe_scale', 1.0)
    wd         = float(args['optimization']['weight_decay'])
    final_wd   = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup     = args['optimization']['warmup']
    start_lr   = args['optimization']['start_lr']
    lr         = args['optimization']['lr']
    final_lr   = args['optimization']['final_lr']

    # ---------------- LOGGING ----------------
    folder = args['logging']['folder']
    tag    = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)

    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    final_path  = os.path.join(folder, f'{tag}-final.pth.tar')
    load_path   = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    csv_logger = CSVLogger(
        os.path.join(folder, f'{tag}.csv'),
        ('%d',   'epoch'),
        ('%.5f', 'loss'),
        ('%.5f', 'jepa_loss'),
        ('%.5f', 'mlm_loss'),
        ('%.4f', 'loss_lambda'),
    )

    # ---------------- MODEL ----------------
    encoder, predictor = init_model(
        device=device,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )

    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    embed_dim  = encoder.token_embed.token_embed.weight.shape[1]
    mlm_head   = MLMHead(embed_dim=embed_dim, vocab_size=vocab_size).to(device)
    loss_weight = nn.Parameter(torch.zeros(1, device=device))

    # ---------------- MASK COLLATOR ----------------
    mask_collator = TextMutiBlockMaskCollector(
        max_tokens=max_tokens,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        enc_mask_scale=enc_mask_scale,
        pred_mask_scale=pred_mask_scale,
        min_keep=min_keep,
        allow_overlap=allow_overlap,
    )

    # ---------------- DATASET ----------------
    loader, sampler = make_textjepa(
        batch_size=batch_size,
        collator=mask_collator,
        num_workers=num_workers,
        max_length=max_seq_len,
        transform=tokenize,
    )

    ipe = len(loader)

    # ---------------- OPTIM ----------------
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )

    optimizer.add_param_group({
        'params': list(mlm_head.parameters()) + [loss_weight],
        'lr': lr,
        'weight_decay': wd,
    })

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0

    if load_model and os.path.exists(load_path):
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, is_final=False):
        save_dict = {
            'encoder':        encoder.state_dict(),
            'predictor':      predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'mlm_head':       mlm_head.state_dict(),
            'loss_weight':    loss_weight.data,
            'opt':            optimizer.state_dict(),
            'scaler':         None if scaler is None else scaler.state_dict(),
            'epoch':          epoch,
            'loss':           loss_meter.avg,
            'batch_size':     batch_size,
            'lr':             lr,
            'config':         args,
        }
        if is_final:
            torch.save(save_dict, final_path)

    # ---------------- TRAIN ----------------
    for epoch in range(start_epoch, num_epochs):

        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        loss_meter      = AverageMeter()
        jepa_loss_meter = AverageMeter()
        mlm_loss_meter  = AverageMeter()
        lambda_meter    = AverageMeter()

        for itr, (tokens, masks_enc, masks_pred) in enumerate(loader):

            try:
                from torch.utils.data._utils.collate import default_collate
            except Exception:
                from torch.utils.data.dataloader import default_collate

            if isinstance(tokens, list):
                tokens = default_collate(tokens)

            tokens = tokens.to(device, non_blocking=True)

            def move_masks_to_device(m):
                if m is None:
                    return None
                if isinstance(m, list) and len(m) > 0 and isinstance(m[0], torch.Tensor):
                    return [t.long().to(device, non_blocking=True) for t in m]
                if isinstance(m, list) and len(m) > 0 and isinstance(m[0], (list, tuple, torch.Tensor)):
                    if isinstance(m[0], torch.Tensor):
                        per_sample = [t.long().to(device, non_blocking=True) for t in m]
                        max_k = max([p.numel() for p in per_sample]) if per_sample else 0
                        if max_k == 0:
                            return []
                        idx_padded = torch.zeros((len(per_sample), max_k), dtype=torch.long, device=device)
                        for i, p in enumerate(per_sample):
                            if p.numel() > 0:
                                idx_padded[i, :p.numel()] = p
                        return [idx_padded]
                    batch_len = len(m)
                    first = m[0]
                    if len(first) > 0 and isinstance(first[0], (list, tuple, torch.Tensor)):
                        n_masks = len(first)
                        out_masks = []
                        for j in range(n_masks):
                            per_sample_indices = []
                            for sample in m:
                                idx_item = sample[j]
                                idx_t = idx_item.long() if isinstance(idx_item, torch.Tensor) \
                                        else torch.tensor(list(idx_item), dtype=torch.long)
                                per_sample_indices.append(idx_t.to(device))
                            max_k = max([p.numel() for p in per_sample_indices]) if per_sample_indices else 0
                            if max_k == 0:
                                out_masks.append(torch.empty((batch_len, 0), dtype=torch.long, device=device))
                                continue
                            idx_padded = torch.zeros((batch_len, max_k), dtype=torch.long, device=device)
                            for i, p in enumerate(per_sample_indices):
                                if p.numel() > 0:
                                    idx_padded[i, :p.numel()] = p
                            out_masks.append(idx_padded)
                        return out_masks
                    else:
                        per_sample_indices = []
                        for sample in m:
                            per_sample_indices.append(
                                sample.long() if isinstance(sample, torch.Tensor)
                                else torch.tensor(list(sample), dtype=torch.long)
                            )
                        max_k = max([p.numel() for p in per_sample_indices]) if per_sample_indices else 0
                        if max_k == 0:
                            return []
                        idx_padded = torch.zeros((batch_len, max_k), dtype=torch.long, device=device)
                        for i, p in enumerate(per_sample_indices):
                            if p.numel() > 0:
                                idx_padded[i, :p.numel()] = p.to(device)
                        return [idx_padded]
                try:
                    return torch.tensor(m, dtype=torch.long, device=device)
                except Exception:
                    return m

            masks_enc  = move_masks_to_device(masks_enc)
            masks_pred = move_masks_to_device(masks_pred)

            def train_step():
                with torch.no_grad():
                    h = target_encoder(tokens)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = tokens.size(0)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    z = encoder(tokens, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    loss_jepa = 1.0 - F.cosine_similarity(z, h, dim=-1).mean()

                masked_tokens, mlm_labels = apply_bert_masking(
                    tokens=tokens,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    mlm_prob=mlm_prob,
                )

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    enc_hidden = encoder(masked_tokens)
                    mlm_logits = mlm_head(enc_hidden)
                    loss_mlm   = F.cross_entropy(
                        mlm_logits.view(-1, vocab_size),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                    )

                lam  = torch.sigmoid(loss_weight)
                loss = lam * loss_jepa + (1.0 - lam) * loss_mlm

                optimizer.zero_grad()
                if use_bfloat16 and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                wd_scheduler.step()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for q, k in zip(encoder.parameters(), target_encoder.parameters()):
                        k.data.mul_(m).add_((1.0 - m) * q.data)

                return loss.item(), loss_jepa.item(), loss_mlm.item(), lam.item()

            loss, loss_jepa, loss_mlm, lam = gpu_timer(train_step)[0]

            loss_meter.update(loss)
            jepa_loss_meter.update(loss_jepa)
            mlm_loss_meter.update(loss_mlm)
            lambda_meter.update(lam)

            if np.isnan(loss) or np.isinf(loss):
                print(f'[Epoch {epoch+1}] NaN/Inf loss — stopping.')
                return

        # ── one print per epoch ───────────────────────────────────────────────
        print(
            f'Epoch [{epoch+1:3d}/{num_epochs}] '
            f'loss={loss_meter.avg:.4f}  '
            f'jepa={jepa_loss_meter.avg:.4f}  '
            f'mlm={mlm_loss_meter.avg:.4f}  '
            f'lambda={lambda_meter.avg:.4f}'
        )

        csv_logger.log(
            epoch + 1,
            loss_meter.avg,
            jepa_loss_meter.avg,
            mlm_loss_meter.avg,
            lambda_meter.avg,
        )

        is_final = (epoch + 1 == num_epochs)
        save_checkpoint(epoch + 1, is_final=is_final)


# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Text-JEPA Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in ['meta', 'data', 'mask', 'optimization', 'logging']:
        if key not in config:
            raise ValueError(f'Missing required config section: {key}')

    main(config, resume_preempt=args.resume)