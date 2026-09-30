from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from aggregation.strategies import (
    TrustAwareFedAvg,
    TrustWeightedRobustTrimming,
    TrustWeightedMedian,
    TrustAwareFedAvgM,
    MultiKrumAggregator
)
from aggregation.candidate_selector import CandidateModelSelector


class AdaptiveAggregator:
    """
    Adaptive Aggregation Engine in TARA-FL.

    Modes:
      1. Default (Risk-based Routing): Dynamically dispatches aggregation based on Round Risk.
         - LOW Risk    -> Trust-Aware FedAvg
         - MEDIUM Risk -> Trust-Weighted Robust Trimming
         - HIGH Risk   -> Trust-Weighted Median
      2. Candidate-Evaluation Mode: Evaluates multiple aggregation strategies in parallel
         against a validation probe set and selects the winning parameter update.
    """

    def __init__(
            self,
            low_threshold: float = 0.30,
            medium_threshold: float = 0.60,
            trim_ratio: float = 0.25,
            trust_floor: float = 0.05,
            probe_dataset: Optional[Dataset] = None
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.trim_ratio = trim_ratio
        self.trust_floor = trust_floor

        # Core aggregation strategy instances
        self.strategy_fedavg = TrustAwareFedAvg(trust_floor=trust_floor)
        self.strategy_robust = TrustWeightedRobustTrimming(trim_ratio=trim_ratio, trust_floor=trust_floor)
        self.strategy_median = TrustWeightedMedian(trust_floor=trust_floor)
        self.strategy_momentum = TrustAwareFedAvgM(trust_floor=trust_floor)
        self.strategy_krum = MultiKrumAggregator(trust_floor=trust_floor)

        self.candidate_selector = CandidateModelSelector(probe_dataset) if probe_dataset is not None else None

    # ======================================================
    # SELECT AGGREGATION STRATEGY
    # ======================================================

    def select_aggregator(self, risk_level: str) -> str:
        if risk_level == "LOW":
            return "TRUST_AWARE_FEDAVG"
        elif risk_level == "MEDIUM":
            return "TRUST_WEIGHTED_ROBUST"
        elif risk_level == "HIGH":
            return "TRUST_WEIGHTED_MEDIAN"
        raise ValueError(f"Unknown risk level: {risk_level}")

    # ======================================================
    # BACKWARD COMPATIBLE DELEGATION METHODS
    # ======================================================

    def calculate_trust_weights(self, client_sizes: List[int], trust_scores: Dict[int, float]) -> List[float]:
        return self.strategy_fedavg.calculate_trust_weights(client_sizes, trust_scores)

    def trust_aware_fedavg(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float]
    ) -> Dict[str, torch.Tensor]:
        return self.strategy_fedavg.aggregate(client_parameters, client_sizes, trust_scores)

    def calculate_update_distances(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, float]:
        return self.strategy_robust._calculate_update_distances(client_updates)

    def select_robust_clients(
            self,
            client_ids: List[int],
            distances: Dict[int, float],
            risk_level: str
    ) -> List[int]:
        if risk_level == "LOW" or not distances:
            return list(client_ids)
        trim_ratio = self.trim_ratio if risk_level == "MEDIUM" else min(0.40, self.trim_ratio * 1.5)
        num_clients = len(client_ids)
        remove_count = min(int(num_clients * trim_ratio), max(0, num_clients - 1))
        if remove_count == 0:
            return list(client_ids)
        sorted_clients = sorted(client_ids, key=lambda cid: distances[cid])
        return sorted_clients[:num_clients - remove_count]

    def trust_weighted_robust(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            client_updates: Dict[int, Dict[str, torch.Tensor]],
            risk_level: str = "MEDIUM"
    ) -> Tuple[Dict[str, torch.Tensor], List[int], Dict[int, float]]:
        distances = self.calculate_update_distances(client_updates)
        client_ids = list(range(1, len(client_parameters) + 1))
        selected_clients = self.select_robust_clients(client_ids, distances, risk_level)

        params = self.strategy_robust.aggregate(
            client_parameters=client_parameters,
            client_sizes=client_sizes,
            trust_scores=trust_scores,
            client_updates=client_updates,
            risk_level=risk_level
        )
        return params, selected_clients, distances

    def trust_weighted_median(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float]
    ) -> Dict[str, torch.Tensor]:
        return self.strategy_median.aggregate(client_parameters, client_sizes, trust_scores)

    # ======================================================
    # MASTER AGGREGATE ROUTER
    # ======================================================

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            risk_level: str = "LOW",
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
            model_template: Optional[nn.Module] = None,
            mode: str = "risk_routing",
            client_ids: Optional[List[int]] = None,
            quarantine_manager: Optional[Any] = None
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Master aggregation dispatch with QuarantineManager support.
        """
        # 1. Resolve client IDs
        if client_ids is None:
            if client_updates is not None and len(client_updates) == len(client_parameters):
                client_ids = list(client_updates.keys())
            elif len(trust_scores) == len(client_parameters):
                client_ids = list(trust_scores.keys())
            else:
                client_ids = list(range(1, len(client_parameters) + 1))

        # 2. Filter quarantined clients if quarantine_manager provided
        if quarantine_manager is not None:
            eligible_indices = []
            eligible_cids = []
            for i, cid in enumerate(client_ids):
                if quarantine_manager.is_eligible_for_aggregation(cid):
                    eligible_indices.append(i)
                    eligible_cids.append(cid)

            if len(eligible_indices) == 0:
                if model_template is not None:
                    return {k: v.clone() for k, v in model_template.state_dict().items()}, "QUARANTINE_ALL_RETAINED"
                return {}, "QUARANTINE_ALL_RETAINED"

            client_parameters = [client_parameters[i] for i in eligible_indices]
            client_sizes = [client_sizes[i] for i in eligible_indices]
            trust_scores = {cid: trust_scores.get(cid, 0.0) for cid in eligible_cids}
            if client_updates is not None:
                client_updates = {cid: client_updates[cid] for cid in eligible_cids}
            client_ids = eligible_cids

        # Mode 2: Multi-Model Candidate Selection
        if mode == "candidate_eval" and self.candidate_selector is not None and model_template is not None:
            best_params, best_strategy, _ = self.candidate_selector.select_best_candidate(
                model_template=model_template,
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_updates=client_updates,
                client_ids=client_ids
            )
            return best_params, best_strategy

        # Mode 1: Risk-based Adaptive Routing
        aggregator_name = self.select_aggregator(risk_level)

        if aggregator_name == "TRUST_AWARE_FEDAVG":
            new_params = self.strategy_fedavg.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_ids=client_ids
            )
        elif aggregator_name == "TRUST_WEIGHTED_ROBUST":
            new_params = self.strategy_robust.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_updates=client_updates,
                risk_level=risk_level,
                client_ids=client_ids
            )
        elif aggregator_name == "TRUST_WEIGHTED_MEDIAN":
            new_params = self.strategy_median.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_ids=client_ids
            )
        else:
            raise ValueError(f"Unsupported aggregator: {aggregator_name}")

        return new_params, aggregator_name