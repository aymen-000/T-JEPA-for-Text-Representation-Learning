"""
Text Masking Strategies for T-JEPA
"""

import random
import torch
from torch.utils.data import default_collate
from multiprocessing import Value
from src.help.utils import sample_block_size, sample_block_mask

class TextMutiBlockMaskCollector:
    """Structured multi-block masking for text (I-JEPA style)"""

    def __init__(self, max_tokens=128, enc_mask_scale=(0.2, 0.5),
                 pred_mask_scale=(0.2, 0.5), nenc=1, npred=2,
                 min_keep=4, allow_overlap=False):
        
        self.max_tokens = max_tokens
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)
        print(f"TextMutiBlockMaskCollector initialized with max_tokens={max_tokens}")

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def __call__(self, batch):
        B = len(batch)
        collated_batch = default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Sample block sizes for predictor and encoder
        p_size = sample_block_size(self.max_tokens, self.pred_mask_scale, generator=g)
        e_size = sample_block_size(self.max_tokens, self.enc_mask_scale, generator=g)

        collated_masks_pred, collated_masks_enc = [], []

        for text_tokens in collated_batch:
            masks_p, masks_c = [], []

            for _ in range(self.npred):
                mask, mask_c = sample_block_mask(text_tokens, p_size, min_keep=self.min_keep)
                masks_p.append(mask)
                masks_c.append(mask_c)

            collated_masks_pred.append(masks_p)

            accep_regions = masks_c if not self.allow_overlap else None
            masks_e = []

            for _ in range(self.nenc):
                mask, _ = sample_block_mask(text_tokens, e_size, min_keep=self.min_keep,
                                            acceptable_regions=accep_regions)
                masks_e.append(mask)
            collated_masks_enc.append(masks_e)

        # Keep masks as lists to handle variable lengths
        return collated_batch, collated_masks_enc, collated_masks_pred


class TextRandomMaskCollector:
    """Random token masking (MAE-style)"""

    def __init__(self, max_tokens=128, ratio=(0.3, 0.6)):
        self.max_tokens = max_tokens
        self.ratio = ratio
        self._itr_counter = Value('i', -1)
        print(f"TextRandomMaskCollector initialized with max_tokens={max_tokens}, ratio={ratio}")

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def __call__(self, batch):
        B = len(batch)
        collated_batch = default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_pred, collated_masks_enc = [], []

        for text_tokens in collated_batch:
            num_tokens = len(text_tokens)
            ratio = self.ratio[0] + torch.rand(1, generator=g).item() * (self.ratio[1] - self.ratio[0])
            num_keep = int(num_tokens * (1.0 - ratio))
            perm = torch.randperm(num_tokens, generator=g)

            collated_masks_enc.append([perm[:num_keep].tolist()])
            collated_masks_pred.append([perm[num_keep:].tolist()])

        # Keep masks as lists for variable-length sequences
        return collated_batch, collated_masks_enc, collated_masks_pred


# --------------------------
# Test script
# --------------------------
if __name__ == "__main__" : 
    # Dummy tokenized text batch
    dummy_batch = [
        torch.arange(20),  # 20 tokens
        torch.arange(15),  # 15 tokens
        torch.arange(30),  # 30 tokens
    ]

    print("="*80)
    print("Testing TextMutiBlockMaskCollector")
    muti_block_masker = TextMutiBlockMaskCollector(max_tokens=30, nenc=1, npred=2, min_keep=3)

    tokens_batch, masks_enc, masks_pred = muti_block_masker(dummy_batch)

    print("Batch size:", len(tokens_batch))
    print("Encoder masks:", masks_enc)
    print("Predictor masks:", masks_pred)

    print("="*80)
    print("Testing TextRandomMaskCollector")
    random_masker = TextRandomMaskCollector(max_tokens=30, ratio=(0.3, 0.5))

    tokens_batch, masks_enc, masks_pred = random_masker(dummy_batch)

    print("Batch size:", len(tokens_batch))
    print("Encoder masks:", masks_enc)
    print("Predictor masks:", masks_pred)

    print("="*80)
    print("All tests completed successfully!")
