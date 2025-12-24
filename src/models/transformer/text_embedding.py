import torch 
import torch.nn as nn 



class TokenEmbed(nn.Module):
    """
    Token Embedding Layer (replaces PatchEmbed for text)
    Converts token IDs to embeddings
    """
    def __init__(self, vocab_size=30522, embed_dim=768, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings 
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Number of "patches" is just sequence length for text
        self.num_patches = max_seq_len
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] - token IDs
        
        Returns:
            embeddings: [batch_size, seq_len, embed_dim]
        """
        return self.token_embed(x)
