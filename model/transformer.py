import torch
import torch.nn as nn
import torch.nn.functional as F

#Has all layers necessary for the transformer model

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
    
#----- DECODER LAYERS -----
class RegularDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size):
        super(RegularDecoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=0.1)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embed_size)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x + self.ff(x)
        return x

class ACTDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size, act_size, halting_threshold, max_thought):
        super(ACTDecoder, self).__init__()
        self.decoder = RegularDecoder(embed_size=embed_size, num_heads=num_heads, ff_size=ff_size)
        self.act = nn.Sequential(
            nn.Linear(embed_size, act_size), 
            nn.ReLU(), 
            nn.Linear(act_size, 1), 
            nn.Sigmoid()
        )
        self.halting_threshold = halting_threshold
        self.max_thought = max_thought
        
    def forward(self, x):
        halting_prob = nn.ParameterList([nn.Parameter(0)])
        unused_prob = nn.Parameter(1)
        outputs = nn.ParameterList()
        
        while (unused_prob >= 1 - self.halting_threshold) and (len(halting_prob) < self.max_thought):
            x = self.decoder(x)
            outputs.append(x)
            
            act = self.act(x)
            
            state = torch.min(act, unused_prob[-1])
            halting_prob.append(nn.Parameter(state))
            unused_prob = unused_prob - state

        remainder = 1 - unused_prob
        #Halting prob starts at 0, but outputs don't
        output = torch.sum(halt * out for halt, out in zip(halting_prob[1:], outputs))
        
        return (output, remainder)