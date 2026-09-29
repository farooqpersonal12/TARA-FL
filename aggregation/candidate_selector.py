from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from aggregation.strategies import (
    TrustAwareFedAvg,
    TrustWeightedRobustTrimming,
    TrustWeightedMedian,
    TrustAwareFedAvgM,
    MultiKrumAggregator
)


class CandidateModelSelector:
    """
    Advanced Multi-Model Candidate Aggregation Evaluator (Briefing p. 14).
    
    Generates candidate models from multiple aggregation strategies simultaneously,
    evaluates their prospective validation performance on a probe set,
    and returns the highest-performing global model.
    """

    def __init__(self, probe_dataset: Dataset, batch_size: int = 32, device: str = "cpu"):
        self.probe_dataset = probe_dataset
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)
        self.loss_func = nn.CrossEntropyLoss()

        self.candidates = {
            "TRUST_AWARE_FEDAVG": TrustAwareFedAvg(),
            "TRUST_WEIGHTED_ROBUST": TrustWeightedRobustTrimming(),
            "TRUST_WEIGHTED_MEDIAN": TrustWeightedMedian(),
            "MULTI_KRUM": MultiKrumAggregator(),
        }

    def _evaluate_model_params(
            self,
            model_template: nn.Module,
            parameters: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """Returns (average_loss, accuracy)."""
        import copy
        try:
            temp_model = copy.deepcopy(model_template).to(self.device)
        except Exception:
            temp_model = type(model_template)().to(self.device)
        temp_model.load_state_dict({k: v.to(self.device) for k, v in parameters.items()})
        temp_model.eval()


        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = temp_model(images)
                loss = self.loss_func(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / max(1, total), correct / max(1, total)

    def select_best_candidate(
            self,
            model_template: nn.Module,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, float]]:
        """
        Runs all candidate aggregators, evaluates them, and returns:
          (best_parameters, best_strategy_name, candidate_eval_scores)
        """
        results = {}
        eval_scores = {}

        for name, aggregator in self.candidates.items():
            params = aggregator.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_updates=client_updates
            )
            loss, acc = self._evaluate_model_params(model_template, params)
            results[name] = params
            eval_scores[name] = acc

        # Pick candidate with highest accuracy (lowest loss as tiebreaker)
        best_name = max(eval_scores, key=lambda k: eval_scores[k])
        return results[best_name], best_name, eval_scores
