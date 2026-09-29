"""
TARA-FL Model Package

Provides base model interfaces, benchmark architectures (MNIST, Fashion-MNIST, CIFAR-10, ResNet, MLP),
and a centralized ModelFactory for dynamic model resolution.
"""

from model.base_model import BaseFLModel
from model.model import (
    MNISTModel,
    FashionMNISTModel,
    CIFAR10ConvNet,
    ResNet18Small,
    SimpleResidualBlock,
    MLPModel,
    ModelFactory,
    get_model,
    list_available_models
)

__all__ = [
    "BaseFLModel",
    "MNISTModel",
    "FashionMNISTModel",
    "CIFAR10ConvNet",
    "ResNet18Small",
    "SimpleResidualBlock",
    "MLPModel",
    "ModelFactory",
    "get_model",
    "list_available_models"
]
