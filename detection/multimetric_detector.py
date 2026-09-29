from typing import Dict, Tuple, Optional
import torch
from detection.base_detector import BaseDetector


class MultiMetricPIDDetector(BaseDetector):
    """
    Advanced Multi-Metric PID Detector.
    
    Combines:
      1. Normalized Euclidean Distance (magnitude difference)
      2. Cosine Dissimilarity (directional angle difference)
      3. Layer-Wise Maximum Disparity (stealthy / targeted layer backdoor tracking)
    """

    def __init__(
            self,
            kp: float = 1.0,
            ki: float = 0.08,
            kd: float = 5.0,
            weight_euclidean: float = 0.50,
            weight_cosine: float = 0.30,
            weight_layerwise: float = 0.20,
            window_size: Optional[int] = None,
            integral_cap: Optional[float] = None
    ):
        super().__init__(name="MultiMetricPIDDetector")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.w_euc = weight_euclidean
        self.w_cos = weight_cosine
        self.w_layer = weight_layerwise
        self.window_size = window_size
        self.integral_cap = integral_cap

    def _calculate_metrics(
            self,
            update: Dict[str, torch.Tensor],
            reference_vector: torch.Tensor,
            reference_dict: Dict[str, torch.Tensor]
    ) -> float:
        update_vector = self._flatten_update(update)

        # 1. Normalized Euclidean Distance
        euc_dist = torch.norm(update_vector - reference_vector, p=2)
        norm_ref = torch.norm(reference_vector, p=2)
        d_euc = (euc_dist / (norm_ref + 1e-12)).item()

        # 2. Cosine Dissimilarity in [0, 2]
        norm_up = torch.norm(update_vector, p=2)
        if norm_up > 1e-12 and norm_ref > 1e-12:
            cos_sim = torch.dot(update_vector, reference_vector) / (norm_up * norm_ref)
            d_cos = max(0.0, min(2.0, (1.0 - cos_sim).item()))
        else:
            d_cos = 0.0

        # 3. Layer-Wise Max Disparity
        layer_disparities = []
        for layer_name in update:
            if torch.is_floating_point(update[layer_name]) and layer_name in reference_dict:
                up_l = update[layer_name].detach().float().flatten()
                ref_l = reference_dict[layer_name].detach().float().flatten()
                l_dist = torch.norm(up_l - ref_l, p=2)
                l_ref = torch.norm(ref_l, p=2)
                layer_disparities.append((l_dist / (l_ref + 1e-12)).item())

        d_layer = max(layer_disparities) if layer_disparities else d_euc

        # Composite Distance
        composite_distance = (
                self.w_euc * d_euc
                + self.w_cos * d_cos
                + self.w_layer * d_layer
        )
        return composite_distance

    def calculate_scores(
            self,
            client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        client_ids = list(client_updates.keys())
        if not client_ids:
            return {}, {}

        # Stack updates and compute median reference
        update_vectors = [self._flatten_update(client_updates[cid]) for cid in client_ids]
        stacked_vectors = torch.stack(update_vectors)
        ref_vector = torch.median(stacked_vectors, dim=0).values

        # Layer-wise median reference dictionary
        ref_dict = {}
        for name in client_updates[client_ids[0]]:
            if torch.is_floating_point(client_updates[client_ids[0]][name]):
                stacked_layer = torch.stack([client_updates[cid][name].detach().float() for cid in client_ids])
                ref_dict[name] = torch.median(stacked_layer, dim=0).values

        distances = {}
        for client_id in client_ids:
            dist = self._calculate_metrics(client_updates[client_id], ref_vector, ref_dict)
            distances[client_id] = dist

        scores = {}
        for client_id in client_ids:
            current_distance = distances[client_id]
            history = self.distance_history.get(client_id, [])
            previous_distance = history[-1] if len(history) > 0 else current_distance

            active_history = history
            if self.window_size is not None and self.window_size > 0:
                active_history = history[-self.window_size:]

            integral = sum(active_history)
            if self.integral_cap is not None:
                integral = min(integral, self.integral_cap)

            derivative = current_distance - previous_distance

            score = max(
                0.0,
                self.kp * current_distance + self.ki * integral + self.kd * derivative
            )
            scores[client_id] = score

            history.append(current_distance)
            self.distance_history[client_id] = history

        return distances, scores
