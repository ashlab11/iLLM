import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.basic_transformer import TransformerLayer

class TestLLM(nn.Module):
    """Test LLM that should perform awfully, but will show proof-of-concept for the training stage"""
    def __init__(self, vocab_size, embed_size, num_mha, num_heads, ff_size):
        super(TestLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_size = ff_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.transformers = nn.ModuleList(
            [TransformerLayer(embed_size, num_heads, ff_size) for _ in range(num_mha)]
        )
        self.lin = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.transformers:
            x = layer(x, causal = True, key_padding_mask = mask)
        if torch.isnan(x).any():
            raise ValueError("Output of transformer layers contains NaN values")
        x = self.lin(x)
        return x
        