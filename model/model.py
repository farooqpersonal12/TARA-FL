import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type, Any, Optional, List

from model.base_model import BaseFLModel


# ==============================================================================
# 1. MNIST MODEL (Backwards-Compatible 4-Layer CNN)
# ==============================================================================

class MNISTModel(BaseFLModel):
    """
    Standard CNN for MNIST digit classification.
    
    Architecture:
        Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
        Flatten
        Linear(1600, 128) -> ReLU
        Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ==============================================================================
# 2. FASHION-MNIST MODEL (Enhanced CNN with Batch Normalization)
# ==============================================================================

class FashionMNISTModel(BaseFLModel):
    """
    Enhanced ConvNet for Fashion-MNIST with Batch Normalization and Dropout.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ==============================================================================
# 3. CIFAR-10 / CIFAR-100 CONVNET
# ==============================================================================

class CIFAR10ConvNet(BaseFLModel):
    """
    3-Stage Convolutional Neural Network designed for 32x32 color images
    (CIFAR-10, CIFAR-100, SVHN).
    
    Architecture:
        Block 1: Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool
        Block 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool
        Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool
        Classifier: Linear(128*4*4, 512) -> ReLU -> Dropout -> Linear(512, num_classes)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: 32x32 -> 16x16
        self.conv1a = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 16x16 -> 8x8
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 8x8 -> 4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Classifier
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ==============================================================================
# 4. LIGHTWEIGHT RESNET (ResNet-18 Small for Edge FL)
# ==============================================================================

class SimpleResidualBlock(nn.Module):
    """Basic Residual Block with shortcut connection."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Small(BaseFLModel):
    """
    Lightweight ResNet tailored for Federated Learning on 32x32 / 28x28 images.
    Has significantly fewer parameters than full ImageNet ResNet-18 while
    preserving skip connections and representation capacity.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, base_planes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, base_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_planes)

        self.layer1 = self._make_layer(base_planes, base_planes, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_planes, base_planes * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_planes * 2, base_planes * 4, num_blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(base_planes * 4, num_classes)

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        curr_in = in_planes
        for s in strides:
            layers.append(SimpleResidualBlock(curr_in, planes, s))
            curr_in = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


# ==============================================================================
# 5. MULTI-LAYER PERCEPTRON (MLP Model for Tabular / Vector Benchmarks)
# ==============================================================================

class MLPModel(BaseFLModel):
    """
    Flexible Multi-Layer Perceptron for 1D tabular or flattened vector inputs.
    """

    def __init__(
            self,
            input_dim: int = 784,
            hidden_dims: Optional[List[int]] = None,
            num_classes: int = 10,
            dropout_rate: float = 0.2
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.network(x)


# ==============================================================================
# 6. MODEL FACTORY & UTILITIES
# ==============================================================================

class ModelFactory:
    """
    Factory for creating and registering Federated Learning model architectures.
    """
    _registry: Dict[str, Type[BaseFLModel]] = {
        "mnist": MNISTModel,
        "fashion_mnist": FashionMNISTModel,
        "fashionmnist": FashionMNISTModel,
        "cifar10": CIFAR10ConvNet,
        "cifar100": CIFAR10ConvNet,
        "cifar_convnet": CIFAR10ConvNet,
        "resnet18": ResNet18Small,
        "resnet": ResNet18Small,
        "mlp": MLPModel,
    }

    @classmethod
    def register(cls, name: str, model_class: Type[BaseFLModel]):
        """Register a new custom model architecture."""
        cls._registry[name.lower().strip()] = model_class

    @classmethod
    def list_models(cls) -> List[str]:
        """Return a sorted list of registered model keys."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseFLModel:
        """
        Instantiate a registered model by name.
        
        Args:
            name: String name of the model (e.g. 'mnist', 'cifar10', 'resnet18', 'mlp').
            **kwargs: Keyword arguments passed to the model's constructor.
        
        Returns:
            An instantiated BaseFLModel.
        """
        key = name.lower().strip()
        if key not in cls._registry:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown model name '{name}'. Registered models are: [{available}]"
            )
        return cls._registry[key](**kwargs)


def get_model(name: str = "mnist", **kwargs) -> BaseFLModel:
    """
    Convenience helper function to retrieve an FL model by name.
    
    Examples:
        model = get_model("mnist")
        model = get_model("cifar10", num_classes=10, in_channels=3)
        model = get_model("resnet18", num_classes=100, in_channels=3)
        model = get_model("mlp", input_dim=784, hidden_dims=[256, 128], num_classes=10)
    """
    return ModelFactory.create(name, **kwargs)


def list_available_models() -> List[str]:
    """Return all available registered model architecture names."""
    return ModelFactory.list_models()