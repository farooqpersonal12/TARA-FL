from typing import Dict, List, Optional, Tuple
import torch
from aggregation.base_aggregator import BaseAggregator


# ==============================================================================
# 1. TRUST-AWARE FEDAVG
# ==============================================================================

class TrustAwareFedAvg(BaseAggregator):
    """Standard FedAvg weighted by client sample size and dynamic trust score."""

    def __init__(self, trust_floor: float = 0.05):
        super().__init__(name="TRUST_AWARE_FEDAVG", trust_floor=trust_floor)

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        normalized_weights = self.calculate_trust_weights(client_sizes, trust_scores)
        new_parameters = {}

        for name in client_parameters[0]:
            new_parameters[name] = torch.zeros_like(client_parameters[0][name])
            for parameters, weight in zip(client_parameters, normalized_weights):
                new_parameters[name] += weight * parameters[name].to(new_parameters[name].device)

        return new_parameters


# ==============================================================================
# 2. TRUST-WEIGHTED ROBUST TRIMMING
# ==============================================================================

class TrustWeightedRobustTrimming(BaseAggregator):
    """
    Robust Trimming Aggregator:
      1. Computes distance of each update from the centroid.
      2. Trims top extreme anomalous client updates.
      3. Aggregates retained subset weighted by size * trust.
    """

    def __init__(self, trim_ratio: float = 0.25, trust_floor: float = 0.05):
        super().__init__(name="TRUST_WEIGHTED_ROBUST", trust_floor=trust_floor)
        self.trim_ratio = trim_ratio

    def _calculate_update_distances(
            self,
            client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, float]:
        if not client_updates:
            return {}
        client_ids = list(client_updates.keys())
        vectors = []
        for client_id in client_ids:
            tensors = [p.detach().float().flatten() for p in client_updates[client_id].values() if torch.is_floating_point(p)]
            vectors.append(torch.cat(tensors))

        stacked = torch.stack(vectors)
        centroid = torch.mean(stacked, dim=0)

        distances = {}
        for client_id, vector in zip(client_ids, vectors):
            distances[client_id] = torch.norm(vector - centroid, p=2).item()
        return distances

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
            risk_level: str = "MEDIUM",
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        num_clients = len(client_parameters)
        client_ids = list(range(1, num_clients + 1))

        if client_updates:
            distances = self._calculate_update_distances(client_updates)
            trim_frac = self.trim_ratio if risk_level == "MEDIUM" else min(0.40, self.trim_ratio * 1.5)
            remove_count = min(int(num_clients * trim_frac), max(0, num_clients - 1))

            if remove_count > 0:
                sorted_clients = sorted(client_ids, key=lambda cid: distances.get(cid, 0.0))
                selected_clients = sorted_clients[:num_clients - remove_count]
            else:
                selected_clients = client_ids
        else:
            selected_clients = client_ids

        # Compute weights for selected clients
        selected_sizes = [client_sizes[cid - 1] for cid in selected_clients]
        normalized_weights = self.calculate_trust_weights(selected_sizes, trust_scores, client_ids=selected_clients)

        new_parameters = {}
        for name in client_parameters[0]:
            new_parameters[name] = torch.zeros_like(client_parameters[0][name])
            for cid, weight in zip(selected_clients, normalized_weights):
                idx = cid - 1
                new_parameters[name] += weight * client_parameters[idx][name].to(new_parameters[name].device)

        return new_parameters


# ==============================================================================
# 3. TRUST-WEIGHTED MEDIAN
# ==============================================================================

class TrustWeightedMedian(BaseAggregator):
    """Coordinate-wise weighted median parameterized by dynamic trust."""

    def __init__(self, trust_floor: float = 0.05):
        super().__init__(name="TRUST_WEIGHTED_MEDIAN", trust_floor=trust_floor)

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        normalized_weights = self.calculate_trust_weights(client_sizes, trust_scores)
        new_parameters = {}

        for name in client_parameters[0]:
            stacked = torch.stack([params[name] for params in client_parameters])
            original_shape = stacked.shape[1:]
            flattened = stacked.reshape(stacked.shape[0], -1)

            # Vectorized coordinate-wise weighted median across client dimension (dim=0)
            sorted_vals, sorted_indices = torch.sort(flattened, dim=0)
            weights_tensor = torch.tensor(normalized_weights, device=stacked.device, dtype=stacked.dtype)
            weight_per_elem = weights_tensor[sorted_indices]
            cum_weights = torch.cumsum(weight_per_elem, dim=0)
            mask = (cum_weights >= 0.5)
            first_true_idx = torch.argmax(mask.int(), dim=0)
            result = torch.gather(sorted_vals, 0, first_true_idx.unsqueeze(0)).squeeze(0)

            new_parameters[name] = result.reshape(original_shape)

        return new_parameters



# ==============================================================================
# 4. TRUST-AWARE FEDAVGM (Server-Side Momentum)
# ==============================================================================

class TrustAwareFedAvgM(BaseAggregator):
    """Server-side momentum FedAvg aggregator."""

    def __init__(self, beta: float = 0.9, server_lr: float = 1.0, trust_floor: float = 0.05):
        super().__init__(name="TRUST_AWARE_FEDAVGM", trust_floor=trust_floor)
        self.beta = beta
        self.server_lr = server_lr
        self.momentum_buffer: Dict[str, torch.Tensor] = {}

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            global_parameters: Optional[Dict[str, torch.Tensor]] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        normalized_weights = self.calculate_trust_weights(client_sizes, trust_scores)

        # 1. Weighted Average Target
        avg_params = {}
        for name in client_parameters[0]:
            avg_params[name] = torch.zeros_like(client_parameters[0][name])
            for p, w in zip(client_parameters, normalized_weights):
                avg_params[name] += w * p[name].to(avg_params[name].device)

        if global_parameters is None:
            return avg_params

        # 2. Server momentum update: delta = avg_params - global_params
        new_params = {}
        for name in avg_params:
            delta = avg_params[name] - global_parameters[name].to(avg_params[name].device)
            if name not in self.momentum_buffer:
                self.momentum_buffer[name] = delta.clone()
            else:
                self.momentum_buffer[name] = self.beta * self.momentum_buffer[name] + delta

            new_params[name] = global_parameters[name].to(avg_params[name].device) + self.server_lr * self.momentum_buffer[name]

        return new_params


# ==============================================================================
# 5. MULTI-KRUM AGGREGATOR
# ==============================================================================

class MultiKrumAggregator(BaseAggregator):
    """Multi-Krum Byzantine-robust aggregator with trust weighting."""

    def __init__(self, num_malicious: int = 1, to_select: Optional[int] = None, trust_floor: float = 0.05):
        super().__init__(name="MULTI_KRUM", trust_floor=trust_floor)
        self.num_malicious = num_malicious
        self.to_select = to_select

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        n = len(client_parameters)
        f = self.num_malicious
        m = self.to_select if self.to_select is not None else max(1, n - f - 2)

        # Flatten parameter vectors
        vectors = []
        for params in client_parameters:
            tensors = [p.detach().float().flatten() for p in params.values() if torch.is_floating_point(p)]
            vectors.append(torch.cat(tensors))

        # Compute pairwise distance matrix
        scores = []
        num_neighbors = max(1, n - f - 2)
        for i in range(n):
            dists = [torch.norm(vectors[i] - vectors[j], p=2).item() for j in range(n) if i != j]
            dists.sort()
            score = sum(dists[:num_neighbors])
            scores.append((score, i))

        scores.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in scores[:m]]
        selected_client_ids = [idx + 1 for idx in selected_indices]
        selected_sizes = [client_sizes[idx] for idx in selected_indices]

        weights = self.calculate_trust_weights(selected_sizes, trust_scores, client_ids=selected_client_ids)

        new_parameters = {}
        for name in client_parameters[0]:
            new_parameters[name] = torch.zeros_like(client_parameters[0][name])
            for idx, w in zip(selected_indices, weights):
                new_parameters[name] += w * client_parameters[idx][name].to(new_parameters[name].device)

        return new_parameters
