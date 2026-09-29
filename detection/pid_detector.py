from typing import Dict, Tuple, Optional
import torch
from detection.base_detector import BaseDetector


class PIDDetector(BaseDetector):
    """
    Standard PID Anomaly Detector using arithmetic mean centroid reference.
    
    Features:
      - Normalized Euclidean distance against mean centroid.
      - Proportional (instantaneous), Integral (cumulative), Derivative (rate of change) terms.
      - Optional sliding window and anti-windup clamping on the integral term.
    """

    def __init__(
            self,
            kp: float = 1.0,
            ki: float = 0.08,
            kd: float = 5.0,
            window_size: Optional[int] = None,
            integral_cap: Optional[float] = None,
            integral_decay: float = 1.0
    ):
        super().__init__(name="PIDDetector")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.window_size = window_size
        self.integral_cap = integral_cap
        self.integral_decay = integral_decay

    def _calculate_distance(
            self,
            update: Dict[str, torch.Tensor],
            centroid: torch.Tensor
    ) -> float:
        update_vector = self._flatten_update(update)
        distance = torch.norm(update_vector - centroid, p=2)
        denominator = torch.norm(centroid, p=2)
        normalized_distance = distance / (denominator + 1e-12)
        return normalized_distance.item()

    def calculate_scores(
            self,
            client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        client_ids = list(client_updates.keys())
        if not client_ids:
            return {}, {}

        update_vectors = []
        for client_id in client_ids:
            vector = self._flatten_update(client_updates[client_id])
            update_vectors.append(vector)

        stacked_updates = torch.stack(update_vectors)
        centroid = torch.mean(stacked_updates, dim=0)

        distances = {}
        for client_id in client_ids:
            distance = self._calculate_distance(client_updates[client_id], centroid)
            distances[client_id] = distance

        scores = {}
        for client_id in client_ids:
            current_distance = distances[client_id]
            history = self.distance_history.get(client_id, [])
            previous_distance = history[-1] if len(history) > 0 else current_distance

            # Compute integral with optional sliding window and decay
            active_history = history
            if self.window_size is not None and self.window_size > 0:
                active_history = history[-self.window_size:]

            if self.integral_decay < 1.0 and active_history:
                weights = [self.integral_decay ** i for i in reversed(range(len(active_history)))]
                integral = sum(d * w for d, w in zip(active_history, weights))
            else:
                integral = sum(active_history)

            # Anti-windup clamping
            if self.integral_cap is not None:
                integral = min(integral, self.integral_cap)

            derivative = current_distance - previous_distance

            score = (
                    self.kp * current_distance
                    + self.ki * integral
                    + self.kd * derivative
            )
            score = max(0.0, score)
            scores[client_id] = score

            history.append(current_distance)
            self.distance_history[client_id] = history

        return distances, scores