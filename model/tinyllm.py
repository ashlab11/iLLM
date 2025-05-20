import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings 
from model.layers import EmbeddingLayer, LowRankEmbedding, TransformerEncoder, ACTEncoder

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
        
        # Encoder layers
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embed_size=embed_size)
                
        self.encoders = nn.ModuleList()
        for _ in range(enc_num_blocks):
            for _ in range(enc_num_prelude_per):
                self.encoders.append(TransformerEncoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size))
            for _ in range(enc_act_per_block):
                self.encoders.append(ACTEncoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size, act_size=embedding_rank, halting_threshold=1e-4, max_thought=25))
        
        self.decoders = nn.Sequential()
        # Decoder layers
        for _ in range(dec_layers):
            self.decoders.append(TransformerEncoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size))
        
        self.lin = nn.Linear(embed_size, vocab_size) # Final linear layer to project the output to vocab size
    
    def encode(self, x_enc):
        """
        Encodes the context using a series of transformer/ACT blocks.
        """
        x = self.embedding(x_enc)
        remainder = nn.Parameter(0)
        for blk in range(self.num_blocks):
            for dec in range(self.num_prelude_per):
                x = self.decoders[blk * (self.num_prelude_per + 1) + dec](x)
            x, r = self.decoders[(blk + 1) * self.num_prelude_per](x)
            remainder += r
        x = self.decoders[-1](x)
        return x, remainder
    
    def forward(self, x, encoder_output):
        # Autoregressive decoding given encoder output as part of the input
        pass
                