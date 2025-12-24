import torch 
import torch.nn as nn 
import numpy as np 
import math 
import logging
import random 
import sys 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger() 


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Applies DropPath (Stochastic Depth) regularization to the input tensor.

    During training, this function randomly drops entire residual paths 
    (i.e., sets the output of certain layers or blocks to zero) with probability `drop_prob`. 
    The remaining paths are scaled by `1 / (1 - drop_prob)` to preserve the expected output.

    Args:
        x (Tensor): Input tensor of shape (B, ...), where B is the batch size.
        drop_prob (float, optional): Probability of dropping a path. Defaults to 0.0.
        training (bool, optional): If True, apply DropPath; otherwise, return input as-is. Defaults to False.

    Returns:
        Tensor: Output tensor after applying DropPath.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor

    return output


def stem_conv(channels , strides  , bias) : 
    stem = []
    for i in range(len(channels) -2) : 
        stem = [nn.Conv2d(channels[i] , channels[i+1] , kernel_size=3 , stride=strides[i] , padding=1 ,bias=bias)]
        
        if not bias : 
            stem += [nn.BatchNorm2d(channels[i+1])] 
            
        stem += [nn.ReLU(inplace=True)] 
        
    return stem 




def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches/tokens), D (feature-dim)]
    :param masks: accepted formats:
        - None -> returns x
        - tensor bool of shape [B, N] -> selects True positions per sample and pads per-row to same length
        - tensor long of shape [B, K] -> indices to keep (same K for all batch)
        - list of M tensors each of shape [B, K] -> treat as M masks, gather per-mask and concat -> [B*M, K, D]
        - list of length B where each element is:
            * list/tuple of indices (single mask per sample)
            * list/tuple of lists -> per-sample multiple masks (e.g., [[i1..],[j1..],...])
      Returns:
        Tensor concatenated across masks similar to previous behavior. Returned tensor is on same device as x.
    """
    if masks is None:
        return x

    B, N, D = x.size()
    device = x.device

    # boolean mask tensor [B, N]
    if isinstance(masks, torch.Tensor):
        if masks.dtype == torch.bool:
            per_row = [torch.nonzero(masks[i], as_tuple=False).squeeze(-1) for i in range(B)]
            # pad to max length
            max_k = max([p.numel() for p in per_row]) if per_row else 0
            if max_k == 0:
                return torch.empty((0, 0, D), device=device)
            idx = torch.zeros((B, max_k), dtype=torch.long, device=device)
            for i, p in enumerate(per_row):
                if p.numel() > 0:
                    idx[i, :p.numel()] = p
            idx = idx.unsqueeze(-1).expand(-1, -1, D)
            return torch.gather(x, 1, idx)

        # long tensor [B, K] -> gather directly (same K across batch)
        masks = masks.long().to(device)
        idx = masks.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(x, 1, idx)
        return gathered if gathered is not None else torch.empty((0, 0, D), device=device)

    # list handling
    if isinstance(masks, list):
        if len(masks) == 0:
            return torch.empty((0, 0, D), device=device)

        first = masks[0]

        # list of tensors -> treat as per-mask tensors [B, K]
        if isinstance(first, torch.Tensor):
            all_x = []
            for m in masks:
                m = m.long().to(device)
                idx = m.unsqueeze(-1).expand(-1, -1, D)
                all_x.append(torch.gather(x, 1, idx))
            # Pad per-mask gathered tensors to same K before concatenation
            max_k = max(t.size(1) for t in all_x) if all_x else 0
            if max_k == 0:
                return torch.empty((0, 0, D), device=device)
            padded = []
            for t in all_x:
                if t.size(1) < max_k:
                    pad_t = torch.zeros((t.size(0), max_k, D), dtype=t.dtype, device=device)
                    pad_t[:, :t.size(1), :] = t
                    padded.append(pad_t)
                else:
                    padded.append(t)
            return torch.cat(padded, dim=0)

        # per-sample list: masks is length B (or should be)
        if isinstance(first, (list, tuple)):
            # Ensure mask list length matches batch size B
            batch_len = len(masks)
            if batch_len != B:
                # broadcast single-mask to all batch items
                if batch_len == 1:
                    masks = masks * B
                    batch_len = B
                else:
                    # if shorter: pad with empty masks; if longer: truncate
                    if batch_len < B:
                        masks = masks + [ [] for _ in range(B - batch_len) ]
                        batch_len = B
                    else:
                        masks = masks[:B]
                        batch_len = B
                    # Note: we deliberately avoid failing hard here because many collators
                    # may produce single-mask or broadcastable masks; warn for visibility
                    logging.warning(f"apply_masks: masks length ({batch_len}) did not match batch size ({B}) - adjusted by padding/truncating")

            # Detect whether first sample contains multiple masks (e.g., [ [m1, m2, ...], ... ])
            first = masks[0]
            inner0 = first[0] if (isinstance(first, (list, tuple)) and len(first) > 0) else None
            multi_per_sample = (inner0 is not None) and (isinstance(inner0, (list, tuple, torch.Tensor)))

            if multi_per_sample and len(first) > 1:
                # multiple masks per sample: iterate masks index j
                n_masks = len(first)
                all_x = []
                for j in range(n_masks):
                    per_sample_indices = []
                    # build list of tensors (one per sample) for mask j
                    for sample in masks:
                        idx_item = sample[j]
                        if isinstance(idx_item, torch.Tensor):
                            idx_t = idx_item.long().to(device)
                        else:
                            idx_t = torch.tensor(idx_item, dtype=torch.long, device=device)
                        per_sample_indices.append(idx_t)
                    max_k = max([p.numel() for p in per_sample_indices]) if per_sample_indices else 0
                    if max_k == 0:
                        all_x.append(torch.empty((B, 0, D), device=device))
                        continue
                    idx_padded = torch.zeros((B, max_k), dtype=torch.long, device=device)
                    for i, p in enumerate(per_sample_indices):
                        if p.numel() > 0:
                            idx_padded[i, :p.numel()] = p
                    idx_expand = idx_padded.unsqueeze(-1).expand(-1, -1, D)
                    gathered = torch.gather(x, 1, idx_expand)
                    all_x.append(gathered)
                return torch.cat(all_x, dim=0)
            else:
                # single mask per sample: masks is [sample0_indices, sample1_indices, ...]
                per_sample_indices = []
                for s in masks:
                    # Case: already a tensor
                    if isinstance(s, torch.Tensor):
                        per_sample_indices.append(s.long().to(device))
                        continue
                    # Case: s is list/tuple
                    if isinstance(s, (list, tuple)):
                        if len(s) == 0:
                            per_sample_indices.append(torch.tensor([], dtype=torch.long, device=device))
                            continue
                        inner = s[0]
                        # If inner is a tensor (e.g. [tensor([...])]) -> use that tensor
                        if isinstance(inner, torch.Tensor):
                            # assume single-mask-per-sample wrapped in list
                            per_sample_indices.append(inner.long().to(device))
                            continue
                        # otherwise assume list of ints -> convert
                        per_sample_indices.append(torch.tensor(list(s), dtype=torch.long, device=device))
                        continue
                    # Fallback: try convert single element
                    try:
                        per_sample_indices.append(torch.tensor(s, dtype=torch.long, device=device))
                    except Exception:
                        # give empty to avoid crashing
                        per_sample_indices.append(torch.tensor([], dtype=torch.long, device=device))

                # At this point len(per_sample_indices) == B (ensured above)
                max_k = max([p.numel() for p in per_sample_indices]) if per_sample_indices else 0
                if max_k == 0:
                    return torch.empty((0, 0, D), device=device)
                idx_padded = torch.zeros((B, max_k), dtype=torch.long, device=device)
                for i, p in enumerate(per_sample_indices):
                    if p.numel() > 0:
                        idx_padded[i, :p.numel()] = p
                idx_expand = idx_padded.unsqueeze(-1).expand(-1, -1, D)
                gathered = torch.gather(x, 1, idx_expand)
                return gathered

    raise ValueError("Unsupported mask format for apply_masks")


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):

        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)


        tensor.uniform_(2 * l - 1, 2 * u - 1)


        tensor.erfinv_()


        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x



"""
Helper functions for text masking
"""

def sample_block_size(max_tokens, scale, generator=None):
    """Sample block size in number of tokens"""
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(random.randint(0, 1e6))
    min_scale, max_scale = scale
    block_size = int(max_tokens * (min_scale + (max_scale - min_scale) * torch.rand(1, generator=generator).item()))
    block_size = max(1, block_size)  # at least 1 token
    return block_size


def sample_block_mask(tokens, block_size, min_keep=1, acceptable_regions=None):
    """
    Generate mask indices for a text sequence
    :param tokens: list/tensor of token IDs
    :param block_size: number of tokens to mask
    :param min_keep: minimum number of tokens to keep
    :param acceptable_regions: list of allowed indices to sample from
    """
    n_tokens = len(tokens)
    if acceptable_regions is None:
        candidate_indices = list(range(n_tokens))
    else:
        # Flatten acceptable regions
        candidate_indices = [i for region in acceptable_regions for i in region]

    if len(candidate_indices) < min_keep:
        block_size = len(candidate_indices)

    start_idx = random.choice(candidate_indices) if candidate_indices else 0
    end_idx = min(start_idx + block_size, n_tokens)
    mask_indices = list(range(start_idx, end_idx))
    masked_context = [i for i in candidate_indices if i not in mask_indices]

    return mask_indices, masked_context

    
    
    
    
def set_seed(seed) : 
    torch.manual_seed(seed)
    np.random.seed(seed)


def config_device() : 
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    return device



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(text):
    return tokenizer(text, truncation=True, padding='max_length',
                     max_length=128, return_tensors='pt')['input_ids'].squeeze(0)





