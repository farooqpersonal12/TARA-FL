from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import torch


class BaseDetector(ABC):
    """
    Abstract Base Class for anomaly and malicious client detectors in TARA-FL.
    """

    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self.distance_history: Dict[int, List[float]] = {}

    @abstractmethod
    def calculate_scores(
            self,
            client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Evaluate client updates and return:
          - distances: Dict[client_id, normalized_distance]
          - anomaly_scores: Dict[client_id, anomaly_score]
        """
        raise NotImplementedError("Subclasses must implement calculate_scores().")

    def _flatten_update(self, update: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten all floating point parameters in a client update dictionary
        into a single 1D vector.
        """
        tensors = []
        for parameter in update.values():
            if torch.is_floating_point(parameter):
                tensors.append(parameter.detach().float().flatten())
        if not tensors:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(tensors)

    def reset_history(self):
        """Clear all historical distance tracking."""
        self.distance_history.clear()

    def get_client_history(self, client_id: int) -> List[float]:
        """Return the distance history for a specific client."""
        return list(self.distance_history.get(client_id, []))
