import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseFLModel(nn.Module, ABC):
    """
    Abstract Base Class for all Federated Learning models in TARA-FL.
    
    Provides standardized utilities for parameter extraction, weight
    synchronization, gradient vector flattening, and model checkpointing.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass must be implemented by all subclasses."""
        raise NotImplementedError("Subclasses must implement forward().")

    def get_weights(self) -> dict:
        """Return a copy of the model state dictionary."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, state_dict: dict):
        """Load state dictionary into the model."""
        self.load_state_dict(state_dict)

    def get_parameter_vector(self) -> torch.Tensor:
        """
        Flatten all floating point model parameters into a single 1D tensor vector.
        Useful for distance calculations, anomaly detection, and clustering.
        """
        tensors = []
        for param in self.parameters():
            if param.requires_grad or torch.is_floating_point(param):
                tensors.append(param.detach().float().flatten())
        if not tensors:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(tensors)

    def set_parameter_vector(self, vector: torch.Tensor):
        """
        Populate model parameters from a flattened 1D tensor vector.
        """
        offset = 0
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad or torch.is_floating_point(param):
                    numel = param.numel()
                    param.copy_(vector[offset:offset + numel].reshape(param.shape))
                    offset += numel

    def get_gradient_vector(self) -> torch.Tensor:
        """
        Flatten all parameter gradients into a single 1D tensor vector.
        Returns zeros for parameters without gradients.
        """
        grads = []
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads.append(param.grad.detach().float().flatten())
                else:
                    grads.append(torch.zeros(param.numel(), dtype=torch.float32, device=param.device))
        if not grads:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(grads)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, filepath: str, extra_meta: dict = None):
        """
        Save model weights and metadata to a checkpoint file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "model_class": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "param_count": self.count_parameters(),
            "meta": extra_meta or {}
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        """
        Load model weights from a checkpoint file.
        Returns the metadata dictionary stored in the checkpoint.
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint.get("meta", {})
