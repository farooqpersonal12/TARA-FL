from typing import Dict, List, Optional, Any, Union
from trustengine.trust_history import TrustHistory
from trustengine.quarantine import QuarantineManager, TrustZone


class TrustEngine:
    """
    Continuous, adaptive trust mechanism.

    Pipeline:
        PID score
            ↓
        Robust relative anomaly (Median / MAD scale)
            ↓
        Current trust (Inverse quadratic decay)
            ↓
        Historical trust (EMA memory)
            ↓
        Persistence penalty (Exponential temporal decay)
            ↓
        Final trust score -> Trust Zones & Quarantine
    """

    def __init__(
            self,
            history_weight: float = 0.6,
            current_weight: float = 0.4,
            persistence_weight: float = 0.2,
            decay_factor: float = 0.8,
            mad_scale: float = 3.0,
            minimum_scale: float = 0.01,
            normal_deviation: float = 1.0,
            quarantine_threshold: float = 0.20,
            enable_quarantine: bool = True,
            required_clean_rounds: int = 2
    ):
        self.history_weight = history_weight
        self.current_weight = current_weight
        self.persistence_weight = persistence_weight
        self.decay_factor = decay_factor
        self.mad_scale = mad_scale
        self.minimum_scale = minimum_scale
        self.normal_deviation = normal_deviation

        self.history = TrustHistory()
        self.quarantine_manager = QuarantineManager(
            quarantine_threshold=quarantine_threshold,
            required_clean_rounds=required_clean_rounds
        ) if enable_quarantine else None

    # ---------------------------------------------------------
    # Robust statistics
    # ---------------------------------------------------------

    def _median(self, values: List[float]) -> float:
        values = sorted(values)
        if not values:
            return 0.0
        n = len(values)
        middle = n // 2
        if n % 2 == 1:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    def _mad(self, values: List[float], median: float) -> float:
        deviations = [abs(float(value) - median) for value in values]
        return self._median(deviations)

    def _calculate_scale(self, values: List[float], median: float, mad: float) -> float:
        """
        Calculate a robust scale.
        Primary: MAD
        Fallback: spread / 4 -> relative scale -> minimum_scale
        """
        if mad > self.minimum_scale:
            return mad
        if not values:
            return self.minimum_scale

        value_range = max(values) - min(values)
        fallback_scale = value_range / 4.0
        if fallback_scale > self.minimum_scale:
            return fallback_scale

        relative_scale = abs(median) * 0.10
        return max(relative_scale, self.minimum_scale)

    # ---------------------------------------------------------
    # Relative anomaly
    # ---------------------------------------------------------

    def calculate_relative_anomaly(self, pid_score: float, all_pid_scores: List[float]) -> float:
        """
        Convert an absolute PID score into a peer-relative anomaly.
        A client at or below the population median receives anomaly = 1.0.
        """
        if not all_pid_scores:
            return 1.0

        scores = [float(score) for score in all_pid_scores]
        score = float(pid_score)

        median = self._median(scores)
        mad = self._mad(scores, median)
        scale = self._calculate_scale(scores, median, mad)

        if score <= median:
            return 1.0

        deviation = (score - median) / scale
        return 1.0 + max(0.0, deviation)

    # ---------------------------------------------------------
    # Current trust
    # ---------------------------------------------------------

    def calculate_current_trust(self, relative_anomaly: float) -> float:
        """
        Convert relative anomaly into continuous current trust.
        """
        anomaly = max(1.0, float(relative_anomaly))
        deviation = anomaly - 1.0

        excess_deviation = max(0.0, deviation - self.normal_deviation)
        if excess_deviation <= 0.0:
            return 1.0

        trust = 1.0 / (1.0 + excess_deviation ** 2)
        return max(0.0, min(1.0, trust))

    # ---------------------------------------------------------
    # Historical behavior
    # ---------------------------------------------------------

    def calculate_historical_trust(self, client_id: int) -> float:
        """Return previous trust score, default 1.0 for new clients."""
        previous_trust = self.history.get_latest_trust(client_id)
        if previous_trust is None:
            return 1.0
        return max(0.0, min(1.0, float(previous_trust)))

    # ---------------------------------------------------------
    # Persistence penalty
    # ---------------------------------------------------------

    def calculate_persistence(self, client_id: int, relative_anomaly: float) -> float:
        """
        Calculate an exponentially decaying persistence penalty for repeated anomalies.
        """
        anomaly = max(1.0, float(relative_anomaly))
        severity = max(0.0, anomaly - 1.0)
        normalized_severity = min(1.0, severity / self.mad_scale)

        anomaly_history = self.history.get_anomaly_history(client_id)
        if not anomaly_history:
            return normalized_severity

        weighted_sum = normalized_severity
        weight_sum = 1.0
        decay_weight = self.decay_factor

        for historical_anomaly in reversed(anomaly_history):
            historical_anomaly = max(1.0, float(historical_anomaly))
            historical_severity = max(0.0, historical_anomaly - 1.0)
            historical_severity = min(1.0, historical_severity / self.mad_scale)

            weighted_sum += historical_severity * decay_weight
            weight_sum += decay_weight
            decay_weight *= self.decay_factor

        return max(0.0, min(1.0, weighted_sum / weight_sum))

    # ---------------------------------------------------------
    # Final trust calculation (Single client & Batch)
    # ---------------------------------------------------------

    def calculate_trust(
            self,
            client_id: Union[int, Dict[int, float]],
            pid_score: Optional[float] = None,
            all_pid_scores: Optional[List[float]] = None
    ) -> Union[Dict[str, float], Dict[int, float]]:
        """
        Calculate trust for one client or batch of clients.
        If client_id is a Dict[int, float] (pid_scores), delegates to calculate_all_trust.
        """
        if isinstance(client_id, dict):
            return self.calculate_all_trust(client_id)

        if pid_score is None or all_pid_scores is None:
            raise ValueError("pid_score and all_pid_scores must be provided when calculating trust for a single client.")
        relative_anomaly = self.calculate_relative_anomaly(pid_score, all_pid_scores)
        current_trust = self.calculate_current_trust(relative_anomaly)
        historical_trust = self.calculate_historical_trust(client_id)
        persistence = self.calculate_persistence(client_id, relative_anomaly)

        final_trust = (
                self.history_weight * historical_trust
                + self.current_weight * current_trust
                - self.persistence_weight * persistence
        )
        final_trust = max(0.0, min(1.0, final_trust))

        # Update quarantine status if enabled
        zone = None
        is_quarantined = False
        if self.quarantine_manager is not None:
            zone = self.quarantine_manager.evaluate_client(
                client_id, final_trust, relative_anomaly
            )
            is_quarantined = self.quarantine_manager.is_quarantined(client_id)
            self.history.record_quarantine_status(client_id, is_quarantined)

        # Store behavior for future rounds
        self.history.update(client_id, final_trust, relative_anomaly)

        return {
            "client_id": client_id,
            "pid_score": float(pid_score),
            "relative_anomaly": float(relative_anomaly),
            "current_trust": float(current_trust),
            "historical_trust": float(historical_trust),
            "persistence": float(persistence),
            "trust": float(final_trust),
            "zone": str(zone) if zone is not None else self.get_trust_zone(final_trust),
            "is_quarantined": is_quarantined,
        }

    def calculate_all_trust(
            self,
            pid_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Batch trust calculation for all clients.
        Returns: Dict[client_id, trust_score]
        """
        all_scores = list(pid_scores.values())
        trust_map = {}
        for client_id, pid_score in pid_scores.items():
            result = self.calculate_trust(client_id, pid_score, all_scores)
            trust_map[client_id] = result["trust"]
        return trust_map

    # ---------------------------------------------------------
    # Trust zone
    # ---------------------------------------------------------

    def get_trust_zone(
            self,
            trust_score: float,
            population_trust: Optional[List[float]] = None
    ) -> str:
        """
        Classify trust relative to the current population.
        """
        trust = float(trust_score)
        if population_trust is None or not population_trust:
            if trust >= 0.75:
                return "NORMAL"
            elif trust >= 0.40:
                return "MEDIUM"
            else:
                return "LOW"

        population_trust = [float(v) for v in population_trust]
        median_trust = self._median(population_trust)
        mad_trust = self._mad(population_trust, median_trust)
        scale = max(mad_trust, self.minimum_scale)

        deviation = (median_trust - trust) / scale
        if deviation >= 3.0:
            return "LOW"
        if deviation >= 1.5:
            return "MEDIUM"
        return "NORMAL"

    def update_history(self, trust_scores: Dict[int, float], anomalies: Optional[Dict[int, float]] = None):
        """
        Explicitly update trust history for all clients if not already recorded.
        """
        for client_id, trust in trust_scores.items():
            anomaly = anomalies.get(client_id, 1.0) if anomalies else 1.0
            # If trust history does not already have an entry for this round or needs update:
            self.history.update(client_id, trust, anomaly)