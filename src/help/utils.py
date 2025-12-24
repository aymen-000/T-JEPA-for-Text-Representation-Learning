import torch 
import torch.nn as nn 
import numpy as np 
import math 
import logging
from random import random 
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
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

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





