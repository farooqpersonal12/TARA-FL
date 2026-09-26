from trustengine.trust_history import TrustHistory


class TrustEngine:
    """
    Continuous, adaptive trust mechanism.

    Pipeline:
        PID score
            ↓
        Robust relative anomaly
            ↓
        Current trust
            ↓
        Historical trust
            ↓
        Persistence penalty
            ↓
        Final trust
    """

    def __init__(
            self,
            history_weight=0.6,
            current_weight=0.4,
            persistence_weight=0.2,
            decay_factor=0.8,
            mad_scale=3.0,
            minimum_scale=0.01,
            normal_deviation=1.0,
    ):
        self.history_weight = history_weight
        self.current_weight = current_weight
        self.persistence_weight = persistence_weight
        self.decay_factor = decay_factor
        self.mad_scale = mad_scale
        self.minimum_scale = minimum_scale
        self.normal_deviation = normal_deviation

        self.history = TrustHistory()

    # ---------------------------------------------------------
    # Robust statistics
    # ---------------------------------------------------------

    def _median(self, values):
        values = sorted(values)

        if not values:
            return 0.0

        n = len(values)
        middle = n // 2

        if n % 2 == 1:
            return float(values[middle])

        return float((values[middle - 1] + values[middle]) / 2.0)

    def _mad(self, values, median):
        deviations = [
            abs(float(value) - median)
            for value in values
        ]

        return self._median(deviations)

    def _calculate_scale(self, values, median, mad):
        """
        Calculate a robust scale.

        Primary:
            MAD

        Fallback:
            spread / 4

        This prevents division by zero when most clients have
        identical PID scores.
        """

        if mad > self.minimum_scale:
            return mad

        if not values:
            return self.minimum_scale

        value_range = max(values) - min(values)

        fallback_scale = value_range / 4.0

        if fallback_scale > self.minimum_scale:
            return fallback_scale

        # If everybody has almost exactly the same score,
        # use a small scale relative to the population value.
        relative_scale = abs(median) * 0.10

        return max(relative_scale, self.minimum_scale)

    # ---------------------------------------------------------
    # Relative anomaly
    # ---------------------------------------------------------

    def calculate_relative_anomaly(self, pid_score, all_pid_scores):
        """
        Convert an absolute PID score into a peer-relative anomaly.

        A client at or below the population median receives
        anomaly = 1.

        Clients above the median receive a continuous anomaly
        proportional to their robust deviation.
        """

        if not all_pid_scores:
            return 1.0

        scores = [
            float(score)
            for score in all_pid_scores
        ]

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

    def calculate_current_trust(self, relative_anomaly):
        """
        Convert relative anomaly into continuous current trust.

        Small deviations inside the normal statistical region
        are treated gently.

        Strong deviations receive increasingly lower trust.
        """

        anomaly = max(1.0, float(relative_anomaly))

        deviation = anomaly - 1.0

        # Normal region:
        # small peer-relative deviations should not heavily
        # penalize otherwise normal clients.
        excess_deviation = max(
            0.0,
            deviation - self.normal_deviation
        )

        if excess_deviation <= 0.0:
            return 1.0

        # Smooth inverse mapping.
        trust = 1.0 / (
                1.0 + excess_deviation ** 2
        )

        return max(0.0, min(1.0, trust))

    # ---------------------------------------------------------
    # Historical behavior
    # ---------------------------------------------------------

    def calculate_historical_trust(self, client_id):
        """
        Return the client's previous trust.

        New clients start with full trust.
        """

        previous_trust = self.history.get_latest_trust(client_id)

        if previous_trust is None:
            return 1.0

        return max(
            0.0,
            min(1.0, float(previous_trust))
        )

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def calculate_persistence(
            self,
            client_id,
            relative_anomaly
    ):
        """
        Calculate a continuous persistence penalty.

        Repeated anomalous behavior contributes more than
        an isolated anomaly.

        Recency weighting is controlled by decay_factor.
        """

        anomaly = max(
            1.0,
            float(relative_anomaly)
        )

        severity = max(
            0.0,
            anomaly - 1.0
        )

        # Convert anomaly severity into [0, 1].
        normalized_severity = min(
            1.0,
            severity / self.mad_scale
        )

        anomaly_history = self.history.get_anomaly_history(
            client_id
        )

        if not anomaly_history:
            return normalized_severity

        weighted_sum = normalized_severity
        weight_sum = 1.0

        decay_weight = self.decay_factor

        for historical_anomaly in reversed(anomaly_history):
            historical_anomaly = max(
                1.0,
                float(historical_anomaly)
            )

            historical_severity = max(
                0.0,
                historical_anomaly - 1.0
            )

            historical_severity = min(
                1.0,
                historical_severity / self.mad_scale
            )

            weighted_sum += (
                    historical_severity * decay_weight
            )

            weight_sum += decay_weight

            decay_weight *= self.decay_factor

        return max(
            0.0,
            min(1.0, weighted_sum / weight_sum)
        )

    # ---------------------------------------------------------
    # Final trust
    # ---------------------------------------------------------

    def calculate_trust(
            self,
            client_id,
            pid_score,
            all_pid_scores
    ):
        """
        Complete trust calculation for one client.
        """

        relative_anomaly = self.calculate_relative_anomaly(
            pid_score,
            all_pid_scores
        )

        current_trust = self.calculate_current_trust(
            relative_anomaly
        )

        historical_trust = self.calculate_historical_trust(
            client_id
        )

        persistence = self.calculate_persistence(
            client_id,
            relative_anomaly
        )

        final_trust = (
                self.history_weight * historical_trust
                + self.current_weight * current_trust
                - self.persistence_weight * persistence
        )

        final_trust = max(
            0.0,
            min(1.0, final_trust)
        )

        # Store behavior for future rounds.
        self.history.update(
            client_id,
            final_trust,
            relative_anomaly
        )

        return {
            "client_id": client_id,
            "pid_score": float(pid_score),
            "relative_anomaly": float(relative_anomaly),
            "current_trust": float(current_trust),
            "historical_trust": float(historical_trust),
            "persistence": float(persistence),
            "trust": float(final_trust),
        }

    # ---------------------------------------------------------
    # Trust zone
    # ---------------------------------------------------------

    def get_trust_zone(
            self,
            trust_score,
            population_trust=None
    ):
        """
        Classify trust relative to the current population.

        This is descriptive only and is NOT used as the primary
        trust calculation.
        """

        trust = float(trust_score)

        if population_trust is None:
            return "NORMAL"

        population_trust = [
            float(value)
            for value in population_trust
        ]

        if not population_trust:
            return "NORMAL"

        median_trust = self._median(
            population_trust
        )

        mad_trust = self._mad(
            population_trust,
            median_trust
        )

        scale = max(
            mad_trust,
            self.minimum_scale
        )

        deviation = (
                            median_trust - trust
                    ) / scale

        if deviation >= 3.0:
            return "LOW"

        if deviation >= 1.5:
            return "MEDIUM"

        return "NORMAL"