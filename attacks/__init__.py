"""
TARA-FL Attacks Package

Provides both data poisoning (label-flip, backdoor) and model/gradient poisoning
(gradient scale, sign flip, gaussian noise, sybil collusion) attacks.
"""

from attacks.base_attack import BaseAttack
from attacks.label_flip import LabelFlipDataset, LabelFlipAttack
from attacks.gradient_poison import GradientScaleAttack, SignFlipAttack
from attacks.backdoor import BackdoorDataset, BackdoorAttack
from attacks.gaussian_noise import GaussianNoiseAttack, RandomParameterZeroAttack
from attacks.sybil_collusion import SybilCollusionCoordinator
from attacks.attack_factory import AttackFactory, get_attack

__all__ = [
    "BaseAttack",
    "LabelFlipDataset",
    "LabelFlipAttack",
    "GradientScaleAttack",
    "SignFlipAttack",
    "BackdoorDataset",
    "BackdoorAttack",
    "GaussianNoiseAttack",
    "RandomParameterZeroAttack",
    "SybilCollusionCoordinator",
    "AttackFactory",
    "get_attack",
]
