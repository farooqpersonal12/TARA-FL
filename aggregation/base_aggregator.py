from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch


class BaseAggregator(ABC):
    """
    Abstract Base Class for Federated Learning Aggregators in TARA-FL.
    """

    def __init__(self, name: str = "BaseAggregator", trust_floor: float = 0.05):
        self.name = name
        self.trust_floor = trust_floor

    @abstractmethod
    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Combine client model parameters into a single global parameter dictionary."""
        raise NotImplementedError("Subclasses must implement aggregate().")

    def calculate_trust_weights(
            self,
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            client_ids: Optional[List[int]] = None
    ) -> List[float]:
        """
        Compute normalized effective weights: weight_k = size_k * max(T_k, trust_floor).
        """
        if client_ids is None:
            client_ids = list(range(1, len(client_sizes) + 1))

        weights = []
        for i, client_id in enumerate(client_ids):
            size = client_sizes[i] if i < len(client_sizes) else client_sizes[client_id - 1]
            trust = trust_scores.get(client_id, 0.0)
            trust = max(0.0, min(1.0, float(trust)))

            effective_trust = max(trust, self.trust_floor) if trust > 0.0 else 0.0
            weights.append(size * effective_trust)

        total_weight = sum(weights)
        if total_weight <= 1e-12:
            total_size = sum(client_sizes[:len(client_ids)])
            return [s / max(1, total_size) for s in client_sizes[:len(client_ids)]]

        return [w / total_weight for w in weights]
