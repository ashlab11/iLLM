import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings 
from transformer import EmbeddingLayer, LowRankEmbedding, RegularDecoder, ACTDecoder

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_blocks, num_prelude_per, 
                 num_heads, ff_size, embedding_rank = 1200, dropout = 0.1):
        super(TinyLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_prelude_per = num_prelude_per
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embed_size=embed_size)
        self.pos_enc = RotaryPositionalEmbeddings(dim = embed_size // num_heads)
                
        self.decoders = nn.ModuleList()
        for _ in range(num_blocks):
            for _ in range(num_prelude_per):
                self.decoders.append(RegularDecoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size))
            self.decoders.append(ACTDecoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size, act_size=embedding_rank, halting_threshold=1e-4, max_thought=10))
        
        self.decoders.append(RegularDecoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size))
        self.lin = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        remainder = nn.Parameter(0)
        for blk in range(self.num_blocks):
            for dec in range(self.num_prelude_per):
                x = self.decoders[blk * (self.num_prelude_per + 1) + dec](x)
            x, r = self.decoders[(blk + 1) * self.num_prelude_per](x)
            remainder += r
        x = self.decoders[-1](x)
        x = self.lin(x)
        return x, remainder
                