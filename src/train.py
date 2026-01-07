import copy
import logging
import sys
import yaml
import os
import numpy as np

import torch
import torch.nn.functional as F

from src.dataset.masks.all_masks import  TextMutiBlockMaskCollector
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
# Logging / reproducibility
# ---------------------------------------------------------
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

log_freq = 10
epoch_log_freq = 10

# ---------------------------------------------------------
def main(args, resume_preempt=False):

    # ---------------- META ----------------
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # ---------------- DATA ----------------
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    vocab_size = args['data'].get('vocab_size', 30522)  # Default BERT vocab
    max_seq_len = args['data'].get('max_seq_len', 512)  # Default max length

    # ---------------- MASK ----------------
    num_enc_masks = args['mask']['num_enc_masks']
    num_pred_masks = args['mask']['num_pred_masks']
    enc_mask_scale = args['mask']['enc_mask_scale']
    pred_mask_scale = args['mask']['pred_mask_scale']
    min_keep = args['mask']['min_keep']
    allow_overlap = args['mask']['allow_overlap']
    max_tokens = args['mask'].get('max_tokens', max_seq_len)  # Use max_seq_len if not specified

    # ---------------- OPT ----------------
    ema = args['optimization']['ema']
    ipe_scale = args['optimization'].get('ipe_scale', 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # ---------------- LOGGING ----------------
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

    # Save config
    config_path = os.path.join(folder, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args, f)
    logger.info(f'Config saved to {config_path}')

    # Checkpoint paths
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    final_path = os.path.join(folder, f'{tag}-final.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # ---------------- CSV ----------------
    csv_logger = CSVLogger(
        os.path.join(folder, f'{tag}.csv'),
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.1f', 'enc_tokens'),
        ('%.1f', 'pred_tokens'),
        ('%d', 'time_ms')
    )

    # ---------------- MODEL ----------------
    logger.info('Initializing models...')
    encoder, predictor = init_model(
        device=device,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )
    
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    pred_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    logger.info(f'Encoder parameters: {enc_params:.2f}M')
    logger.info(f'Predictor parameters: {pred_params:.2f}M')

    # ---------------- MASK COLLATOR ----------------
    # For text, use a simpler mask collator or adapt TextMutiBlockMaskCollector
    # Here we assume TextMutiBlockMaskCollector works with 1D sequences
    logger.info('Initializing mask collator...')
    mask_collator = TextMutiBlockMaskCollector(
        max_tokens=max_tokens,  # For text: sequence length
        nenc=num_enc_masks,
        npred=num_pred_masks,
        enc_mask_scale=enc_mask_scale,
        pred_mask_scale=pred_mask_scale,
        min_keep=min_keep,
        allow_overlap=allow_overlap
    )

    # ---------------- DATASET ----------------
    logger.info('Loading dataset...')
    loader, sampler = make_textjepa(
            batch_size=batch_size,
            collator=mask_collator,
            num_workers=num_workers,
            max_length=max_seq_len, 
            transform=tokenize
        )

    
    ipe = len(loader)
    logger.info(f'Dataset loaded: {ipe} iterations per epoch')

    # ---------------- OPTIM ----------------
    logger.info('Initializing optimizer...')
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
        use_bfloat16=use_bfloat16
    )

    # Momentum scheduler for EMA
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0

    # Load checkpoint if resuming
    if load_model and os.path.exists(load_path):
        logger.info(f'Loading checkpoint from {load_path}')
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler
        )
        # Fast-forward schedulers
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()
        logger.info(f'Resumed from epoch {start_epoch}')

    # Save checkpoint function
    def save_checkpoint(epoch, is_final=False):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr,
            'config': args
        }
        
        
        # Save final model
        if is_final:
            torch.save(save_dict, final_path)
            logger.info(f'✓ Final model saved to {final_path}')

    # ---------------- TRAIN ----------------
    logger.info('='*80)
    logger.info('Starting Text-JEPA training')
    logger.info('='*80)
    logger.info(f'Total epochs: {num_epochs}')
    logger.info(f'Iterations per epoch: {ipe}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Logging every {log_freq} iterations')
    logger.info(f'Detailed logging every {epoch_log_freq} epochs')
    logger.info('='*80)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        # Set epoch for distributed sampler (if applicable)
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        enc_meter = AverageMeter()
        pred_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (tokens, masks_enc, masks_pred) in enumerate(loader):

            # Normalize tokens: collator may return list of tensors -> default_collate into tensor [B, L]
            try:
                from torch.utils.data._utils.collate import default_collate
            except Exception:
                from torch.utils.data.dataloader import default_collate

            if isinstance(tokens, list):
                tokens = default_collate(tokens)

            # Send tokens to device
            tokens = tokens.to(device, non_blocking=True)

            # Helper to move nested mask structures to device and convert per-sample lists
            # into per-mask padded tensors [B, K] (predictor expects list of per-mask tensors).
            def move_masks_to_device(m):
                if m is None:
                    return None
                # Already list of per-mask tensors: [M] tensors of shape [B, K]
                if isinstance(m, list) and len(m) > 0 and isinstance(m[0], torch.Tensor):
                    return [t.long().to(device, non_blocking=True) for t in m]
                # Per-sample list: length B, each element either:
                #  - list/tuple of indices (single mask per sample)
                #  - list/tuple of lists (multiple masks per sample)
                if isinstance(m, list) and len(m) > 0 and isinstance(m[0], (list, tuple, torch.Tensor)):
                    # If per-sample elements are tensors (single-mask-per-sample), convert to cpu tensors list first
                    if isinstance(m[0], torch.Tensor):
                        per_sample = [t.long().to(device, non_blocking=True) for t in m]
                        # treat as single-mask-per-sample => build one per-mask tensor
                        max_k = max([p.numel() for p in per_sample]) if per_sample else 0
                        if max_k == 0:
                            return []
                        idx_padded = torch.zeros((len(per_sample), max_k), dtype=torch.long, device=device)
                        for i, p in enumerate(per_sample):
                            if p.numel() > 0:
                                idx_padded[i, :p.numel()] = p
                        return [idx_padded]
                    # Now each element is a Python list/tuple
                    batch_len = len(m)
                    first = m[0]
                    # Detect multiple masks per sample (e.g. sample -> [m1, m2, ...])
                    if len(first) > 0 and isinstance(first[0], (list, tuple, torch.Tensor)):
                        n_masks = len(first)
                        out_masks = []
                        for j in range(n_masks):
                            per_sample_indices = []
                            for sample in m:
                                idx_item = sample[j]
                                if isinstance(idx_item, torch.Tensor):
                                    idx_t = idx_item.long()
                                else:
                                    idx_t = torch.tensor(list(idx_item), dtype=torch.long)
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
                        # single mask per sample: build one [B, K] tensor
                        per_sample_indices = []
                        for sample in m:
                            if isinstance(sample, torch.Tensor):
                                per_sample_indices.append(sample.long())
                            else:
                                per_sample_indices.append(torch.tensor(list(sample), dtype=torch.long))
                        max_k = max([p.numel() for p in per_sample_indices]) if per_sample_indices else 0
                        if max_k == 0:
                            return []
                        idx_padded = torch.zeros((batch_len, max_k), dtype=torch.long, device=device)
                        for i, p in enumerate(per_sample_indices):
                            if p.numel() > 0:
                                idx_padded[i, :p.numel()] = p.to(device)
                        return [idx_padded]
                # Fallback: try convert to tensor on device
                try:
                    return torch.tensor(m, dtype=torch.long, device=device)
                except Exception:
                    return m

            masks_enc = move_masks_to_device(masks_enc)
            masks_pred = move_masks_to_device(masks_pred)

            # Compute average mask lengths for logging (handles both per-mask tensors and per-sample lists)
            def avg_mask_len(m):
                if m is None:
                    return 0.0
                if isinstance(m, list) and len(m) > 0:
                    if isinstance(m[0], torch.Tensor):
                        # per-mask tensors: shape [B, K] -> use K of first mask
                        try:
                            return float(m[0].size(1))
                        except Exception:
                            return float(m[0].numel() / max(1, m[0].size(0)))
                    # per-sample list
                    if isinstance(m[0], (list, tuple)):
                        # multiple masks per sample: average length of first inner mask across batch
                        lengths = []
                        for sample in m:
                            if isinstance(sample, (list, tuple)) and len(sample) > 0:
                                first = sample[0]
                                lengths.append(len(first) if not isinstance(first, torch.Tensor) else int(first.numel()))
                            else:
                                lengths.append(0)
                        return float(sum(lengths) / max(1, len(lengths)))
                    # per-sample single tensor list
                    if isinstance(m[0], torch.Tensor):
                        return float(sum(int(x.numel()) for x in m) / max(1, len(m)))
                # fallback
                return 0.0

            enc_meter.update(avg_mask_len(masks_enc))
            pred_meter.update(avg_mask_len(masks_pred))

            def train_step():
                # Forward target encoder (no gradients)
                with torch.no_grad():
                    h = target_encoder(tokens)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = tokens.size(0)
                    # Apply target masks
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

                # Forward encoder and predictor
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    z = encoder(tokens, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    loss = 1 - F.cosine_similarity(z, h, dim=-1).mean()

                # Backward pass
                optimizer.zero_grad()
                if use_bfloat16 and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Step schedulers (moved outside the if/else)
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # Get gradient stats
                grad_stats = grad_logger(encoder.named_parameters())
                
                with torch.no_grad():
                    z_std = z.std(dim=0).mean().item()
                    h_std = h.std(dim=0).mean().item()

                logger.info(f"z_std={z_std:.4f}, h_std={h_std:.4f}")

                # EMA update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for q, k in zip(encoder.parameters(), target_encoder.parameters()):
                        k.data.mul_(m).add_((1 - m) * q.data)

                return (loss.item(), _new_lr, _new_wd, grad_stats)

            # Time the training step
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # Logging
            if ((epoch + 1) % epoch_log_freq == 0 or epoch == 0) and (itr % log_freq == 0):
                logger.info(
                    f'[{epoch+1:3d}, {itr:5d}] '
                    f'loss: {loss_meter.avg:.4f} | '
                    f'enc_tok: {enc_meter.avg:.1f} | '
                    f'pred_tok: {pred_meter.avg:.1f} | '
                    f'lr: {_new_lr:.2e} | '
                    f'wd: {_new_wd:.2e} | '
                    f'mem: {torch.cuda.max_memory_allocated() / 1024.**2 if torch.cuda.is_available() else 0:.0f}MB | '
                    f'time: {time_meter.avg:.1f}ms'
                )


                
                if grad_stats is not None:
                    logger.info(
                        f'[{epoch+1:3d}, {itr:5d}] '
                        f'grad: [{grad_stats.first_layer:.2e}, {grad_stats.last_layer:.2e}] '
                        f'range: ({grad_stats.min:.2e}, {grad_stats.max:.2e})'
                    )

            # CSV logging (every iteration)
            csv_logger.log(
                epoch + 1, itr,
                loss_meter.avg,
                enc_meter.avg,
                pred_meter.avg,
                int(time_meter.avg)
            )

            # Check for NaN
            if np.isnan(loss) or np.isinf(loss):
                logger.error(f'NaN or Inf loss detected at epoch {epoch+1}, iteration {itr}')
                logger.error('Stopping training')
                return

        # End of epoch
        logger.info(f'Epoch {epoch + 1}/{num_epochs} completed | avg_loss: {loss_meter.avg:.4f}')
        
        # Save checkpoint
        is_final = (epoch + 1 == num_epochs)
        save_checkpoint(epoch + 1, is_final=is_final)

    # Training complete
    logger.info('='*80)
    logger.info('TEXT-JEPA TRAINING COMPLETE!')
    logger.info('='*80)
    logger.info(f'Final loss: {loss_meter.avg:.4f}')
    logger.info(f'Final model saved to: {final_path}')
    logger.info(f'Latest checkpoint: {latest_path}')
    logger.info(f'Training logs: {csv_logger.fname}')
    logger.info('='*80)


# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Text-JEPA Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Validate config
    required_keys = ['meta', 'data', 'mask', 'optimization', 'logging']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required config section: {key}')

    main(config, resume_preempt=args.resume)