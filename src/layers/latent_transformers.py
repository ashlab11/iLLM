import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.basic_transformer import TransformerLayer

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