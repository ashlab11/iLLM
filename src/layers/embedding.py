import torch
import torch.nn as nn
import torch.nn.functional as F

#----- EMBEDDING LAYERS -----
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)
    
#Will want to see how this performs vs. regular embedding
class LowRankEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, rank):
        super(LowRankEmbedding, self).__init__()
        self.A = nn.Embedding(vocab_size, rank)
        self.B = nn.Linear(rank, embed_size)

    def forward(self, x):
        return self.B(self.A(x))
