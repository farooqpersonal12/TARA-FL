from typing import Dict, Type, Any, List
from attacks.base_attack import BaseAttack
from attacks.label_flip import LabelFlipAttack
from attacks.gradient_poison import GradientScaleAttack, SignFlipAttack
from attacks.backdoor import BackdoorAttack
from attacks.gaussian_noise import GaussianNoiseAttack, RandomParameterZeroAttack


class AttackFactory:
    """Factory to construct adversarial attack instances by name."""

    _registry: Dict[str, Type[BaseAttack]] = {
        "label_flip": LabelFlipAttack,
        "gradient_scale": GradientScaleAttack,
        "sign_flip": SignFlipAttack,
        "backdoor": BackdoorAttack,
        "gaussian_noise": GaussianNoiseAttack,
        "param_zero": RandomParameterZeroAttack,
    }

    @classmethod
    def register(cls, name: str, attack_class: Type[BaseAttack]):
        cls._registry[name.lower().strip()] = attack_class

    @classmethod
    def list_attacks(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAttack:
        key = name.lower().strip()
        if key not in cls._registry:
            available = ", ".join(cls.list_attacks())
            raise ValueError(f"Unknown attack '{name}'. Available: [{available}]")
        return cls._registry[key](**kwargs)


def get_attack(name: str, **kwargs) -> BaseAttack:
    """Convenience helper to retrieve an attack instance."""
    return AttackFactory.create(name, **kwargs)
