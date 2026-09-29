from abc import ABC, abstractmethod
from typing import Any


class BaseAttack(ABC):
    """
    Abstract Base Class for adversarial attacks in Federated Learning.
    
    Supports both:
      - Data poisoning attacks (modifying client dataset)
      - Model / gradient poisoning attacks (modifying client updates)
    """

    def __init__(self, name: str = "BaseAttack"):
        self.name = name

    @abstractmethod
    def apply(self, target: Any) -> Any:
        """Apply the attack to a dataset or model update dictionary."""
        raise NotImplementedError("Subclasses must implement apply().")
