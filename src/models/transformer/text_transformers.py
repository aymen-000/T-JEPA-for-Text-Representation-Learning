import math
import numpy as np 
from functools import partial

import torch 
import torch.nn as nn 
from src.models.transformer.pos_encode import get_1d_sincos_pos_embed
from src.models.transformer.mlp import Block
from src.help.utils import *
from src.models.transformer.text_embedding import TokenEmbed



class TransformerPredictor(nn.Module):
    """
    Text Transformer Predictor 
    Predicts masked tokens from context
    """
    
    def __init__(self, 
                 max_seq_len=512,
                 embed_dim=768,
                 pred_embed_dim=384,
                 depth=6,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 init_std=0.02,
                 attn_drop_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Project encoder embeddings to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, pred_embed_dim, bias=True)
        
        # Mask token (like [MASK] in BERT)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        
        # Dropout scheduling
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 1D Positional embeddings (sequence positions)
        self.pred_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, pred_embed_dim), 
            requires_grad=False
        )
        
        # Initialize with sinusoidal positions
        pred_pos_embed = get_1d_sincos_pos_embed(pred_embed_dim, max_seq_len)
        self.pred_pos_embed.data.copy_(torch.from_numpy(pred_pos_embed).float().unsqueeze(0))
        
        # Transformer blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=pred_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_scale=qk_scale,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        
        self.predictor_norm = norm_layer(pred_embed_dim)
        self.predictor_proj = nn.Linear(pred_embed_dim, embed_dim, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
            
        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=self.init_std)
                
    def forward(self, x, masks_x, masks):
        """
        Args:
            x: [B*len(masks_x), N_context, D] - context token embeddings
            masks_x: List of context masks (visible tokens)
            masks: List of target masks (tokens to predict)
        
        Returns:
            predictions: [B*len(masks), N_target, D] - predicted embeddings
        """
        assert masks is not None
        assert masks_x is not None
        
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
            
        if not isinstance(masks, list):
            masks = [masks]
            
        B = len(x) // len(masks_x)
        
        # Project to predictor dimension
        x = self.predictor_embed(x)
        
        # Add positional embeddings to context
        x_pos_embed = self.pred_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)
        
        _, N_ctxt, D = x.shape
        
        # Positional embeddings for target (masked) positions
        pos_embed = self.pred_pos_embed.repeat(B, 1, 1)
        pos_embed = apply_masks(pos_embed, masks)
        pos_embed = repeat_interleave_batch(pos_embed, B, repeat=len(masks_x))
        
        # Create mask tokens with positional info
        pred_tokens = self.mask_token.repeat(pos_embed.size(0), pos_embed.size(1), 1)
        pred_tokens += pos_embed
        
        # Concatenate context and mask tokens
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        
        # Apply transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        
        x = self.predictor_norm(x)
        
        # Extract only the predicted tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        
        return x


class TextTransformer(nn.Module):
    """
    Text Transformer Encoder (replaces VisionTransformer)
    Standard Transformer architecture for text processing
    """
    
    def __init__(
        self,
        vocab_size=30522,      # BERT vocab size (can change)
        max_seq_len=512,       # Maximum sequence length
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Token embedding layer (replaces patch embedding)
        self.token_embed = TokenEmbed(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len
        )
        
        # 1D Positional embeddings (sequence positions)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim), 
            requires_grad=False
        )
        pos_embed = get_1d_sincos_pos_embed(embed_dim, max_seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=self.init_std)

    def forward(self, x, masks=None):
        """
        Args:
            x: [batch_size, seq_len] - token IDs
            masks: List of masks indicating visible tokens
        
        Returns:
            embeddings: [batch_size, N_visible, embed_dim]
        """
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Token embeddings
        x = self.token_embed(x)  # [B, seq_len, embed_dim]
        B, N, D = x.shape

        # Add positional embeddings
        # Truncate pos_embed to actual sequence length
        pos_embed = self.pos_embed[:, :N, :]
        x = x + pos_embed

        # Apply masks (for context tokens)
        if masks is not None:
            x = apply_masks(x, masks)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


# ============================================================================
# Model Factory Functions
# ============================================================================

def text_predictor(**kwargs):
    """Create text predictor"""
    model = TransformerPredictor(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def text_transformer_tiny(vocab_size=30522, max_seq_len=512, **kwargs):
    """Tiny text transformer (5M params)"""
    model = TextTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def text_transformer_small(vocab_size=30522, max_seq_len=512, **kwargs):
    """Small text transformer (22M params)"""
    model = TextTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def text_transformer_base(vocab_size=30522, max_seq_len=512, **kwargs):
    """Base text transformer (86M params) - similar to BERT-base"""
    model = TextTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def text_transformer_large(vocab_size=30522, max_seq_len=512, **kwargs):
    """Large text transformer (340M params) - similar to BERT-large"""
    model = TextTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# Embedding dimensions lookup
TEXT_TRANSFORMER_EMBED_DIMS = {
    'text_transformer_tiny': 192,
    'text_transformer_small': 384,
    'text_transformer_base': 768,
    'text_transformer_large': 1024,
}
        
    
         
# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TEXT-JEPA: Text Transformer Example")
    print("="*80)
    
    # Create a tiny text transformer
    encoder = text_transformer_tiny(vocab_size=400, max_seq_len=128)
    predictor = text_predictor(
        max_seq_len=128,
        embed_dim=192,
        pred_embed_dim=96,
        depth=6,
        num_heads=3
    )
    
    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")
    print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M")
    
    # Example forward pass
    batch_size = 4
    seq_len = 128
    
    # Random token IDs (simulating tokenized text)
    input_ids = torch.randint(0, 400, (batch_size, seq_len))
    
    # Example masks (context: first 100 tokens, target: last 28 tokens)
    context_mask = [torch.arange(100)]
    target_mask = [torch.arange(100, 128)]
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Context tokens: {len(context_mask[0])}")
    print(f"Target tokens: {len(target_mask[0])}")
    
    # Forward pass
    with torch.no_grad():
        # Encoder: process context tokens
        h = encoder(input_ids, masks=context_mask)
        print(f"\nEncoder output: {h.shape}")
        
        # Predictor: predict target tokens
        z = predictor(h, masks_x=context_mask, masks=target_mask)
        print(f"Predictor output: {z.shape}")
    
    print("\n" + "="*80)
    print("✓ Text-JEPA working correctly!")
    print("="*80)