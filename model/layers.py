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

#----- TRANSFORMER LAYERS -----
class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=0.1)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embed_size)
        )

    def forward(self, x, causal=False):
        if self.training and causal:
            #Apply causal attention mask if in training model and causal is True
            attn_output, _ = self.attention(x, x, x, is_causal=causal, attn_mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool())
        else:
            attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x + self.ff(x)
        x = F.layer_norm(x, x.size()[1:], eps=1e-6)
        return x

class ACTEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_transformers, ff_size, act_size, halting_threshold, max_thought):
        super(ACTEncoder, self).__init__()
        self.transformers = nn.Sequential(
            *[TransformerLayer(embed_size, num_heads, ff_size) for _ in range(num_transformers)]
        )
        self.act = nn.Sequential(
            nn.Linear(embed_size + 1, act_size), 
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
        thoughts = 0
        
        while (unused_prob >= 1 - self.halting_threshold) and (len(halting_prob) < self.max_thought):
            x = self.decoder(x)
            outputs.append(x)
            
            thoughts += 1
            
            #Add a dimension to x noting how many thoughts have been used -- theoretically this will impact 
            #Dimension of x is (batch_size, seq_len, embed_size)
            x = torch.cat((x, torch.full((x.size(0), 1), thoughts).to(x.device)), dim=2)
            act = self.act(x)
            x = x[:, :, :-1]
            
            state = torch.min(act, unused_prob[-1])
            halting_prob.append(nn.Parameter(state))
            unused_prob = unused_prob - state

        remainder = 1 - unused_prob
        #Halting prob starts at 0, but outputs don't
        output = torch.sum(halt * out for halt, out in zip(halting_prob[1:], outputs))
        
        return (output, remainder)