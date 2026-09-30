from typing import Dict, Tuple, Optional, Any
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
            distances: Optional[Dict[int, float]] = None,
            trust_scores: Optional[Dict[int, float]] = None,
            current_accuracy: Optional[float] = None,
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> Tuple[float, str, int]:
        """
        Calculate composite round risk score and risk level.
        
        Args:
            distances: Pairwise / peer anomaly distance map {client_id: distance}.
            trust_scores: Current trust scores {client_id: trust}.
            current_accuracy: Latest validation accuracy for performance drop detection.
            client_updates: Raw parameter update vectors for geometric threat classification.

        Returns:
            (risk_score, risk_level, suspicious_clients)
        """
        has_distances = bool(distances)
        has_trust = bool(trust_scores)

        if not has_distances and not has_trust:
            self.last_risk_score = 0.0
            self.last_risk_level = "LOW"
            self.last_suspicious_clients = 0
            self.last_threat_type = ThreatType.CLEAN
            self.last_details = {
                "risk_score": 0.0,
                "risk_level": "LOW",
                "suspicious_clients": 0,
                "threat_type": ThreatType.CLEAN.value,
                "anomaly_component": 0.0,
                "trust_component": 0.0,
                "suspicious_ratio": 0.0,
                "worst_client_component": 0.0,
                "accuracy_drop_penalty": 0.0
            }
            return 0.0, "LOW", 0

        # 1. Average Anomaly Component
        if has_distances:
            average_distance = sum(distances.values()) / len(distances)
            anomaly_component = average_distance / (1.0 + average_distance)
        else:
            anomaly_component = 0.0

        # 2. Trust Deficit & Distribution Components
        if has_trust:
            average_trust = sum(trust_scores.values()) / len(trust_scores)
            trust_component = 1.0 - average_trust

            suspicious_clients = sum(
                1 for trust in trust_scores.values() if trust < self.susp_thresh
            )
            suspicious_ratio = suspicious_clients / len(trust_scores)

            minimum_trust = min(trust_scores.values())
            worst_client_component = 1.0 - minimum_trust
        else:
            trust_component = 0.0
            suspicious_clients = 0
            suspicious_ratio = 0.0
            worst_client_component = 0.0

        # 3. Combined Round Risk with available signal normalization
        if has_distances and has_trust:
            risk_score = (
                    self.w_anom * anomaly_component
                    + self.w_trust * trust_component
                    + self.w_susp * suspicious_ratio
                    + self.w_worst * worst_client_component
            )
        elif has_trust:
            total_w = self.w_trust + self.w_susp + self.w_worst
            risk_score = (
                    self.w_trust * trust_component
                    + self.w_susp * suspicious_ratio
                    + self.w_worst * worst_client_component
            ) / max(1e-6, total_w)
        else:
            risk_score = anomaly_component

        # 4. Optional Performance Degradation Feedback
        acc_drop_penalty = 0.0
        if self.enable_perf and current_accuracy is not None and self.previous_accuracy is not None:
            if self.previous_accuracy > 0.0:
                acc_drop = max(0.0, (self.previous_accuracy - current_accuracy) / self.previous_accuracy)
                # Escalate risk if validation accuracy experienced sharp drop (>5%)
                if acc_drop > 0.05:
                    acc_drop_penalty = 0.20 * acc_drop
                    risk_score = min(1.0, risk_score + acc_drop_penalty)

        if current_accuracy is not None:
            self.previous_accuracy = current_accuracy

        risk_score = max(0.0, min(1.0, float(risk_score)))

        # Determine Risk Level
        if risk_score < self.low_risk_threshold:
            risk_level = "LOW"
        elif risk_score < self.medium_risk_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Classify threat signature
        threat_type = self.threat_classifier.classify_threat(
            distances=distances or {},
            trust_scores=trust_scores or {},
            client_updates=client_updates
        )

        self.last_risk_score = risk_score
        self.last_risk_level = risk_level
        self.last_suspicious_clients = suspicious_clients
        self.last_threat_type = threat_type
        self.last_details = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "suspicious_clients": suspicious_clients,
            "threat_type": threat_type.value,
            "anomaly_component": anomaly_component,
            "trust_component": trust_component,
            "suspicious_ratio": suspicious_ratio,
            "worst_client_component": worst_client_component,
            "accuracy_drop_penalty": acc_drop_penalty
        }

        return (
            risk_score,
            risk_level,
            suspicious_clients
        )

    def classify_threat(
            self,
            distances: Optional[Dict[int, float]] = None,
            trust_scores: Optional[Dict[int, float]] = None,
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> ThreatType:
        """Identify the dominant attack signature in this round."""
        return self.threat_classifier.classify_threat(
            distances=distances or {},
            trust_scores=trust_scores or {},
            client_updates=client_updates
        )

    def get_last_details(self) -> Dict[str, Any]:
        """Return full diagnostic breakdown of the most recent risk assessment."""
        return getattr(self, "last_details", {})