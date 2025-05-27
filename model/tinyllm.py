import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import EmbeddingLayer, TransformerEncoder, ACTEncoder

class TestLLM(nn.Module):
    """Test LLM that should perform awfully, but will show proof-of-concept for the training stage"""
    def __init__(self, vocab_size, embed_size, num_mha, num_heads, ff_size):
        super(TestLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_size = ff_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, ff_size), num_mha
        )
        self.lin = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, tgt_is_causal = True)
        x = self.lin(x)
        return x
        

class TinyLLM(nn.Module):
    # Creates a model that includes both encoder and decoder layers, wkth encoder including ACT layers.
    def __init__(
        self, vocab_size, embed_size, 
        enc_num_blocks, enc_num_prelude_per, enc_act_per_block, dec_layers, 
        num_heads, ff_size, embedding_rank = 1200, dropout = 0.1
        ):
        """TinyLLM model with ACT layers in the encoder and transformer layers in the decoder.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the embedding.
            enc_num_blocks (int): Number of blocks (each containing prelude and recurrent ) in the encoder.
            enc_num_prelude_per (int): Number of prelude layers per block in the encoder.
            enc_recur_per_block (int): Number of recurrent layers per block in the encoder.
            dec_layers (int): Number of layers in the decoder.
            num_heads (int): Number of attention heads.
            ff_size (int): Size of the feedforward layer.
            embedding_rank (int, optional): Rank for low-rank embedding. Defaults to 1200.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(TinyLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.enc_num_blocks = enc_num_blocks
        self.enc_num_prelude_per = enc_num_prelude_per
        self.enc_act_per_block = enc_act_per_block
        self.dec_layers = dec_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        
        # Embedding layer is shared between encoder and decoder
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embed_size=embed_size)  
        
        # -- ENCODER LAYERS --
        self.encoders = nn.ModuleList()
        for _ in range(enc_num_blocks):
            for _ in range(enc_num_prelude_per):
                self.encoders.append(TransformerEncoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size))
            self.encoders.append(ACTEncoder(embed_size=embed_size, num_heads=num_heads, num_transformers=enc_act_per_block, ff_size=ff_size, act_size=embedding_rank, halting_threshold=1e-4, max_thought=25))
        self.encoders.append(TransformerEncoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size))
        
        # -- DECODER LAYERS --
        self.decoders = nn.ModuleList()
        # Here we add transformer layers including cross-attention
        
        self.lin = nn.Linear(embed_size, vocab_size) # Final linear layer to project the output to vocab size
    
    def encode(self, x_enc):
        """
        Encodes the context using a series of transformer/ACT blocks.
        """
        x = self.embedding(x_enc)
        remainder = nn.Parameter(0)
        for blk in range(self.num_blocks):
            for enc in range(self.num_prelude_per):
                x = self.encoders[blk * (self.num_prelude_per + 1) + enc](x)
            x, r = self.encoders[(blk + 1) * self.num_prelude_per](x)
            remainder += r
        x = self.encoders[-1](x)
        return x, remainder
    
    def forward(self, x, encoder_output):
        # Autoregressive decoding given encoder output as part of the input
        pass