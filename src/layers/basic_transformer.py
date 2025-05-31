import torch
import torch.nn as nn
import torch.nn.functional as F

#----- TRANSFORMER LAYERS -----
class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embed_size)
        )

    def forward(self, x, causal=False, key_padding_mask=None):
        if causal:
            # If causal is True, we apply a causal attention mask
            # This mask ensures that the model does not attend to future tokens
            causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        else:
            causal_mask = None
        
        attn_output, _ = self.attention(x, x, x, is_causal = causal, key_padding_mask=key_padding_mask, attn_mask = causal_mask)
        x = x + attn_output
        x = F.layer_norm(x, x.size()[1:], eps=1e-6)
        x = x + self.ff(x)
        x = F.layer_norm(x, x.size()[1:], eps=1e-6)
        return x
    
class SwappedTransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size):
        super(SwappedTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embed_size)
        )

    def forward(self, x, causal=False, key_padding_mask=None):
        if causal:
            causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        else:
            causal_mask = None
        
        attn_output, _ = self.attention(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2), is_causal = causal, key_padding_mask=key_padding_mask, attn_mask = causal_mask)
        x = x + attn_output.transpose(1, 2)
        x = F.layer_norm(x, x.size()[1:], eps=1e-6)
        x = x + self.ff(x)
        x = F.layer_norm(x, x.size()[1:], eps=1e-6)
        return x