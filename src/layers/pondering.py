import torch
import torch.nn as nn
from abc import abstractmethod

# ---- Each of these takes in a module and returns a "thinking" version of that module ----
class Planner(nn.Module):
    def __init__(self, module, pool = 'mean'):
        super(Planner, self).__init_()
        self.module = module
        self.embed_size = module.embed_size
        if pool not in {"mean", "cls"}:
            raise ValueError("pool must be 'mean' or 'cls'")
        self.pool = pool
    
    def _pool(self, h):
        """Collapse sequence dimension before passing to Lambda."""
        if self.pool == "mean":
            return h.mean(dim=1)              # [B, D]
        else:
            return h[:, 0]                    # [B, D]   (first token)
        
    @abstractmethod
    def forward(self, x):
        """Returns output (o_utput, extra loss)"""
        pass
    
    
# --- Uses Adaptive Computation Time (ACT) ---
class ACTPlanner(Planner):
    def __init__(self, module, act_size, halting_threshold, max_thought):
        super(ACTPlanner, self).__init__()
        self.module = module
        self.act = nn.Sequential(
            nn.Linear(module.embed_size + 1, act_size), 
            nn.ReLU(), 
            nn.Linear(act_size, 1), 
            nn.Sigmoid()
        )
        self.halting_threshold = halting_threshold
        self.max_thought = max_thought
    
    def forward(self, x):
        halting_prob = nn.ParameterList([nn.Parameter(torch.tensor(0.0))])
        unused_prob = nn.Parameter(torch.tensor(1.0))
        outputs = nn.ParameterList()
        thoughts = 0
        
        while (unused_prob >= 1 - self.halting_threshold) and (len(halting_prob) < self.max_thought):
            x = self.module(x)
            outputs.append(x)
            
            thoughts += 1
            
            # Add a dimension to x noting how many thoughts have been used
            x = torch.cat((x, torch.full((x.size(0), 1), thoughts).to(x.device)), dim=2)
            act = self.act(x)
            x = x[:, :, :-1]
            
            state = torch.min(act, unused_prob[-1])
            halting_prob.append(nn.Parameter(state))
            unused_prob = unused_prob - state

        remainder = 1 - unused_prob
        output = torch.sum(halt * out for halt, out in zip(halting_prob[1:], outputs))
        
        return (output, remainder)


# --- Uses PALBERT-like methods ---
class PALBERTPlanner(nn.Module):
    def __init__(self, module, pool = 'mean', lambda_hidden = 128, q_threshold = 0.5, max_steps = 12, rho_prior = 0.5):
        super(PALBERTPlanner, self).__init__(module, pool)
        self.module = module
        self.embed_size = module.embed_size
        self.lambda_hidden = lambda_hidden
        self.q_threshold = q_threshold
        self.rho_prior = rho_prior
        self.max_steps = max_steps
        self.pool = pool
        
        # Lambda network: [h_i , h_{i-1} , step_id]  → λ_i  ∈ (0,1)
        self.lambda_net = nn.Sequential(
            nn.Linear(self.embed_size * 2 + 1, lambda_hidden),
            nn.Tanh(),
            nn.Linear(lambda_hidden, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        pass
