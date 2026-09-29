from typing import Optional, Union, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.base_model import BaseFLModel
from model.model import MNISTModel, get_model


class Client:
    """
    Federated Learning Edge Client.
    
    Features:
      - Dynamic model architecture injection (via BaseFLModel instance or model name string).
      - Configurable optimizers (SGD, Adam, AdamW) with learning rate and momentum/weight decay.
      - Device placement (CPU / GPU CUDA / Apple MPS).
      - Local Differential Privacy (LDP) via gradient norm clipping and calibrated Gaussian noise.
      - Local evaluation & metric reporting.
    """

    def __init__(
            self,
            client_id: int,
            dataset: Dataset,
            model: Optional[Union[BaseFLModel, nn.Module, str]] = None,
            lr: float = 0.01,
            batch_size: int = 32,
            optimizer_type: str = "sgd",
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            device: Optional[str] = None,
            enable_dp: bool = False,
            dp_clip_norm: float = 1.0,
            dp_noise_multiplier: float = 0.01
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type.lower().strip()
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Local Differential Privacy
        self.enable_dp = enable_dp
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier

        # Model initialization
        if model is None:
            self.model = MNISTModel()
        elif isinstance(model, str):
            self.model = get_model(model)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError(f"Invalid model type: {type(model)}")

        self.model.to(self.device)

    def set_model(self, parameters: Dict[str, torch.Tensor]):
        """Load global parameters into local model."""
        clean_params = {k: v.to(self.device) for k, v in parameters.items()}
        self.model.load_state_dict(clean_params)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer instance based on configuration."""
        if self.optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

    def train(self, epochs: int = 1) -> nn.Module:
        """
        Execute local training on the client's dataset.
        """
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_function = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer()

        self.model.train()

        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_function(outputs, labels)
                loss.backward()

                # Local Differential Privacy: Gradient Clipping + Gaussian Noise
                if self.enable_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.dp_clip_norm
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * (
                                    self.dp_noise_multiplier * self.dp_clip_norm
                            )
                            param.grad.add_(noise)

                optimizer.step()

        return self.model

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[float, float]:
        """
        Evaluate local model on local or provided validation dataset.
        Returns: (average_loss, top1_accuracy)
        """
        dataset = eval_dataset if eval_dataset is not None else self.dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        loss_func = nn.CrossEntropyLoss()
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = loss_func(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return avg_loss, accuracy

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return cloned model state dict transferred to CPU."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def get_update(self, global_parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate and return delta update = local_param - global_param."""
        update = {}
        local_parameters = self.get_parameters()

        for name in local_parameters:
            global_tensor = global_parameters[name].detach().cpu()
            update[name] = local_parameters[name] - global_tensor

        return update