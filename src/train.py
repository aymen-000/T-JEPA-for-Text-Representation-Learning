import copy
import logging
import sys
import yaml
import os
import numpy as np

import torch
import torch.nn.functional as F

from src.dataset.masks.all_masks import MutiBlockMaskCollector
from src.help.utils import apply_masks, repeat_interleave_batch
from src.help.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter
)
from dataset.data.text_data import make_textjepa
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

    # ---------------- MASK ----------------
    num_enc_masks = args['mask']['num_enc_masks']
    num_pred_masks = args['mask']['num_pred_masks']
    enc_mask_scale = args['mask']['enc_mask_scale']
    pred_mask_scale = args['mask']['pred_mask_scale']
    min_keep = args['mask']['min_keep']
    allow_overlap = args['mask']['allow_overlap']
    max_tokens = args['mask']['max_tokens']

    # ---------------- OPT ----------------
    ema = args['optimization']['ema']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    ipe_scale = args['optimization']['ipe_scale']

    # ---------------- LOGGING ----------------
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

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
    encoder, predictor = init_model(
        device=device,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # ---------------- MASK COLLATOR ----------------
    mask_collator = MutiBlockMaskCollector(
        max_tokens=max_tokens,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        enc_mask_scale=enc_mask_scale,
        pred_mask_scale=pred_mask_scale,
        min_keep=min_keep,
        allow_overlap=allow_overlap
    )

    # ---------------- DATASET ----------------
    loader, sampler = make_textjepa(
        batch_size=batch_size,
        collator=mask_collator,
        num_workers=num_workers
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
        use_bfloat16=use_bfloat16
    )

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0

    # ---------------- TRAIN ----------------
    logger.info('Starting Text-JEPA training')

    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        loss_meter = AverageMeter()
        enc_meter = AverageMeter()
        pred_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (tokens, masks_enc, masks_pred) in enumerate(loader):

            tokens = tokens.to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            enc_meter.update(len(masks_enc[0][0]))
            pred_meter.update(len(masks_pred[0][0]))

            def train_step():
                scheduler.step()
                wd_scheduler.step()

                with torch.no_grad():
                    h = target_encoder(tokens)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, tokens.size(0), len(masks_enc))

                z = encoder(tokens, masks_enc)
                z = predictor(z, masks_enc, masks_pred)

                loss = F.smooth_l1_loss(z, h)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for q, k in zip(encoder.parameters(), target_encoder.parameters()):
                        k.data.mul_(m).add_((1 - m) * q.data)

                return loss.item()

            loss, etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            if itr % log_freq == 0:
                csv_logger.log(
                    epoch + 1, itr,
                    loss_meter.avg,
                    enc_meter.avg,
                    pred_meter.avg,
                    int(time_meter.avg)
                )

        logger.info(f'Epoch {epoch + 1} | loss={loss_meter.avg:.4f}')

    logger.info('Training complete')


# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config, resume_preempt=args.resume)
