import numpy as np

def get_1d_sincos_pos_embed(embed_dim, seq_len):
    """
    Generate 1D sinusoidal positional embeddings for text sequences
    
    Args:
        embed_dim: Embedding dimension
        seq_len: Sequence length
    
    Returns:
        pos_embed: [seq_len, embed_dim]
    """
    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * 
                      -(np.log(10000.0) / embed_dim))
    
    pos_embed = np.zeros((seq_len, embed_dim), dtype=np.float32)
    pos_embed[:, 0::2] = np.sin(position * div_term)
    pos_embed[:, 1::2] = np.cos(position * div_term)
    
    return pos_embed

