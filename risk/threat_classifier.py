from enum import Enum
from typing import Dict, List, Optional
import torch


class ThreatType(str, Enum):
    """Classified environmental threat types."""
    CLEAN = "CLEAN"
    MAGNITUDE_SCALING = "MAGNITUDE_SCALING"
    SIGN_FLIP_ANGULAR = "SIGN_FLIP_ANGULAR"
    TARGETED_POISONING = "TARGETED_POISONING"
    NON_IID_DRIFT = "NON_IID_DRIFT"


class ThreatClassifier:
    """
    Analyzes update geometries and statistical deviations to classify
    the primary attack signature in the current federated round.
    """

    def __init__(self, magnitude_threshold: float = 3.0, angular_threshold: float = 1.2):
        self.magnitude_threshold = magnitude_threshold
        self.angular_threshold = angular_threshold

    def classify_threat(
            self,
            distances: Optional[Dict[int, float]] = None,
            trust_scores: Optional[Dict[int, float]] = None,
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> ThreatType:
        if not distances and not trust_scores:
            return ThreatType.CLEAN

        distances = distances or {}
        trust_scores = trust_scores or {}

        max_dist = max(distances.values()) if distances else 0.0
        min_trust = min(trust_scores.values()) if trust_scores else 1.0
        avg_dist = sum(distances.values()) / len(distances) if distances else 0.0

        # 1. Clean condition
        if max_dist < 1.0 and min_trust >= 0.75:
            return ThreatType.CLEAN

        # 2. Check for extreme magnitude scaling
        if max_dist >= self.magnitude_threshold:
            return ThreatType.MAGNITUDE_SCALING

        # 3. Check for angular divergence (Sign-Flip) if raw updates are provided
        if client_updates is not None and len(client_updates) >= 2:
            try:
                # Compute centroid on CPU
                vectors = []
                for cid in client_updates:
                    tensors = [p.detach().float().cpu().flatten() for p in client_updates[cid].values() if torch.is_floating_point(p)]
                    if tensors:
                        vectors.append(torch.cat(tensors))
                if len(vectors) >= 2:
                    stacked = torch.stack(vectors)
                    centroid = torch.mean(stacked, dim=0)

                    # Check if any client has negative cosine similarity with centroid
                    for v in vectors:
                        norm_v = torch.norm(v, p=2)
                        norm_c = torch.norm(centroid, p=2)
                        if norm_v > 1e-12 and norm_c > 1e-12:
                            cos_sim = (torch.dot(v, centroid) / (norm_v * norm_c)).item()
                            if cos_sim < -0.1:
                                return ThreatType.SIGN_FLIP_ANGULAR
            except Exception:
                pass

        # 4. Moderate deviation with isolated low-trust -> Targeted Poisoning
        if min_trust < 0.40 and (max_dist >= 1.5 or not distances):
            return ThreatType.TARGETED_POISONING

        # 5. Mild dispersed deviation with acceptable trust -> Non-IID Drift
        return ThreatType.NON_IID_DRIFT
