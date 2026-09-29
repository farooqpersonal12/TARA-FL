from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ValidationProbeEvaluator:
    """
    Evaluates client updates directly on a clean server-side validation probe dataset.
    
    Computes loss reduction:
      Delta_Loss_k = Loss(w_global) - Loss(w_global + Delta_w_k)
    
    Positive Delta_Loss indicates constructive/beneficial update (High Quality).
    Negative Delta_Loss indicates destructive/poisonous update (Low Quality).
    """

    def __init__(self, probe_dataset: Dataset, batch_size: int = 32, device: str = "cpu"):
        self.probe_dataset = probe_dataset
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)
        self.loss_func = nn.CrossEntropyLoss()

    def _compute_loss(self, model: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.loss_func(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
        return total_loss / max(1, total_samples)

    def evaluate_updates(
            self,
            global_model: nn.Module,
            client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, float]:
        """
        Evaluate all client updates and return functional quality scores in [0, 1].
        """
        global_loss = self._compute_loss(global_model)
        quality_scores = {}

        for client_id, update in client_updates:
            # Apply prospective update to cloned model state
            prospective_state = {}
            for name, param in global_model.state_dict().items():
                if name in update:
                    prospective_state[name] = param + update[name].to(param.device)
                else:
                    prospective_state[name] = param.clone()

            # Temp model for loss check
            temp_model = type(global_model)().to(self.device)
            temp_model.load_state_dict(prospective_state)
            prospective_loss = self._compute_loss(temp_model)

            delta_loss = global_loss - prospective_loss
            # Continuous sigmoid-like mapping to [0, 1]
            quality = 1.0 / (1.0 + float(torch.exp(torch.tensor(-delta_loss * 10.0)).item()))
            quality_scores[client_id] = max(0.0, min(1.0, quality))

        return quality_scores
