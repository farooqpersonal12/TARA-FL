from typing import Dict, Tuple, Optional
import torch
from risk.threat_classifier import ThreatClassifier, ThreatType


class RoundRisk:
    """
    Environmental Round-Risk Assessment Engine.

    Calculates a multi-factor composite risk score in [0, 1] considering:
      1. Average peer anomaly distance (R_anom)
      2. Population trust deficit (R_trust)
      3. Suspicious client ratio (R_susp)
      4. Worst-client extreme metric (R_worst)
      5. Optional validation performance degradation penalty (R_perf)
    """

    def __init__(
            self,
            low_risk_threshold: float = 0.30,
            medium_risk_threshold: float = 0.60,
            weight_anom: float = 0.25,
            weight_trust: float = 0.25,
            weight_suspicious: float = 0.20,
            weight_worst: float = 0.30,
            suspicious_trust_threshold: float = 0.75,
            enable_performance_feedback: bool = True
    ):
        self.low_risk_threshold = low_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.w_anom = weight_anom
        self.w_trust = weight_trust
        self.w_susp = weight_suspicious
        self.w_worst = weight_worst
        self.susp_thresh = suspicious_trust_threshold
        self.enable_perf = enable_performance_feedback

        self.threat_classifier = ThreatClassifier()
        self.previous_accuracy: Optional[float] = None

    def calculate_risk(
            self,
            distances: Dict[int, float],
            trust_scores: Dict[int, float],
            current_accuracy: Optional[float] = None,
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> Tuple[float, str, int]:
        """
        Calculate composite round risk score and risk level.
        
        Returns:
            (risk_score, risk_level, suspicious_clients)
        """
        if not distances or not trust_scores:
            return 0.0, "LOW", 0

        # 1. Average Anomaly Component
        average_distance = sum(distances.values()) / len(distances)
        anomaly_component = average_distance / (1.0 + average_distance)

        # 2. Average Trust Deficit Component
        average_trust = sum(trust_scores.values()) / len(trust_scores)
        trust_component = 1.0 - average_trust

        # 3. Suspicious Client Component
        suspicious_clients = sum(
            1 for trust in trust_scores.values() if trust < self.susp_thresh
        )
        suspicious_ratio = suspicious_clients / len(trust_scores)

        # 4. Worst Client Component
        minimum_trust = min(trust_scores.values())
        worst_client_component = 1.0 - minimum_trust

        # 5. Combined Round Risk
        risk_score = (
                self.w_anom * anomaly_component
                + self.w_trust * trust_component
                + self.w_susp * suspicious_ratio
                + self.w_worst * worst_client_component
        )

        # 6. Optional Performance Degradation Feedback
        if self.enable_perf and current_accuracy is not None and self.previous_accuracy is not None:
            if self.previous_accuracy > 0.0:
                acc_drop = max(0.0, (self.previous_accuracy - current_accuracy) / self.previous_accuracy)
                # Escalate risk if validation accuracy experienced sharp drop (>5%)
                if acc_drop > 0.05:
                    risk_score = min(1.0, risk_score + 0.20 * acc_drop)

        if current_accuracy is not None:
            self.previous_accuracy = current_accuracy

        risk_score = max(0.0, min(1.0, risk_score))

        # Determine Risk Level
        if risk_score < self.low_risk_threshold:
            risk_level = "LOW"
        elif risk_score < self.medium_risk_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return (
            risk_score,
            risk_level,
            suspicious_clients
        )

    def classify_threat(
            self,
            distances: Dict[int, float],
            trust_scores: Dict[int, float],
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> ThreatType:
        """Identify the dominant attack signature in this round."""
        return self.threat_classifier.classify_threat(distances, trust_scores, client_updates)