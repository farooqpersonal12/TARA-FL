import torch
from typing import Dict
from attacks.base_attack import BaseAttack


class GradientScaleAttack(BaseAttack):
    """
    Gradient / model-update poisoning attack.

    After the malicious client performs normal local training,
    the update delta is scaled by a large factor.

    This pushes the global model toward the attacker's
    local gradient direction with disproportionate influence.

    Usage:
        attack = GradientScaleAttack(scale_factor=10.0)
        poisoned_update = attack.apply(client_update)
    """

    def __init__(self, scale_factor: float = 10.0):
        super().__init__(name="GradientScaleAttack")
        self.scale_factor = scale_factor

    def apply(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Scale every parameter in the update by scale_factor.
        """
        poisoned_update = {}
        for name, delta in update.items():
            if torch.is_floating_point(delta):
                poisoned_update[name] = delta * self.scale_factor
            else:
                poisoned_update[name] = delta.clone()
        return poisoned_update


class SignFlipAttack(BaseAttack):
    """
    Sign-flip attack.

    Reverses the direction of the client update so it
    moves the global model away from convergence.

    This is harder to detect than simple scaling because
    the update magnitude can remain similar to honest clients.

    Usage:
        attack = SignFlipAttack()
        poisoned_update = attack.apply(client_update)
    """

    def __init__(self, scale_factor: float = 1.0):
        super().__init__(name="SignFlipAttack")
        self.scale_factor = scale_factor

    def apply(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Negate every parameter in the update.
        """
        poisoned_update = {}
        for name, delta in update.items():
            if torch.is_floating_point(delta):
                poisoned_update[name] = -1.0 * self.scale_factor * delta
            else:
                poisoned_update[name] = delta.clone()
        return poisoned_update
