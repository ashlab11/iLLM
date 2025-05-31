"""
This script provides a list of sampling strategies for text generation using a language model.
"""

import torch
from abc import ABC, abstractmethod

class Sampler(ABC):
    """Abstract base class for sampling strategies."""
    def construct_probabilities(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Convert logits to probabilities using softmax with temperature."""
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")
        return torch.softmax(logits / temperature, dim=-1)
    
    @abstractmethod
    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample a token from the logits."""
        pass

class GreedySampler(Sampler):
    """Greedy sampling strategy."""
    def __init__(self):
        super().__init__()
    
    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """Select the token with the highest probability."""
        return int(logits.argmax().item())
    
class TopKSampler(Sampler):
    """Top-k sampling strategy."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample from the top-k tokens."""
        if self.k <= 0:
            raise ValueError("k must be greater than 0.")
        top_k_logits, indices = torch.topk(logits, self.k)
        probabilities = self.construct_probabilities(top_k_logits, temperature)
        return int(indices[torch.multinomial(probabilities, num_samples=1)].item())
    
class MinPSampler(Sampler):
    """Minimum probability sampling strategy."""
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio
    
    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample from the logits with a minimum probability threshold."""
        probabilities = self.construct_probabilities(logits, temperature)
        max_p = probabilities.max()
        min_p = max_p * self.ratio
        probabilities[probabilities < min_p] = 0
        probabilities /= probabilities.sum()  # Normalizing to ensure valid distribution
        return int(torch.multinomial(probabilities, num_samples=1).item())