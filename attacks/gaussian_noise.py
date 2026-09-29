from typing import Dict
import torch
from attacks.base_attack import BaseAttack


class GaussianNoiseAttack(BaseAttack):
    """
    Adds zero-mean Gaussian noise directly to model parameter updates.
    
    Simulates malicious parameter corruption, noisy channel degradation,
    or random perturbation attacks.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int = 42):
        super().__init__(name="GaussianNoiseAttack")
        self.mean = mean
        self.std = std
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

    def apply(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poisoned_update = {}
        for name, delta in update.items():
            if torch.is_floating_point(delta):
                noise = torch.randn(
                    delta.shape,
                    generator=self.generator,
                    dtype=delta.dtype,
                    device=delta.device
                ) * self.std + self.mean
                poisoned_update[name] = delta + noise
            else:
                poisoned_update[name] = delta.clone()
        return poisoned_update


class RandomParameterZeroAttack(BaseAttack):
    """
    Randomly zeros out a fraction of parameter updates (dropout attack).
    """

    def __init__(self, zero_ratio: float = 0.5, seed: int = 42):
        super().__init__(name="RandomParameterZeroAttack")
        self.zero_ratio = zero_ratio
        self.generator = torch.Generator().manual_seed(seed)

    def apply(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poisoned_update = {}
        for name, delta in update.items():
            if torch.is_floating_point(delta):
                mask = torch.rand(
                    delta.shape,
                    generator=self.generator,
                    device=delta.device
                ) > self.zero_ratio
                poisoned_update[name] = delta * mask.float()
            else:
                poisoned_update[name] = delta.clone()
        return poisoned_update
