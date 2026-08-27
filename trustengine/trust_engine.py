from trustengine.trust_history import TrustHistory


class TrustEngine:

    def __init__(
            self,
            history_weight=0.7,
            current_weight=0.3
    ):

        self.history_weight = history_weight
        self.current_weight = current_weight

        self.history = TrustHistory(
            initial_trust=1.0
        )

    def calculate_relative_anomaly(
            self,
            scores
    ):
        """
        Calculate how anomalous each client is
        relative to the median behavior of the clients.

        A value close to 1.0 means the client is
        close to normal peer behavior.

        A value greater than 1.0 means the client
        is more anomalous than the peer baseline.
        """

        if not scores:
            return {}

        values = sorted(scores.values())

        n = len(values)

        if n % 2 == 0:

            median_score = (
                                   values[n // 2 - 1]
                                   + values[n // 2]
                           ) / 2

        else:

            median_score = values[n // 2]

        median_score = max(
            median_score,
            1e-12
        )

        relative_anomaly = {}

        for client_id, score in scores.items():

            relative_anomaly[client_id] = (
                    score / median_score
            )

        return relative_anomaly

    def calculate_current_trust(
            self,
            anomaly
    ):
        """
        Convert relative anomaly into continuous
        current trust evidence.

        Relative anomaly <= 1.0:
            Client is at or below peer median.

        Relative anomaly > 1.0:
            Client is more anomalous than peer median.

        Trust is bounded between 0 and 1.
        """

        current_trust = {}

        for client_id, anomaly_value in anomaly.items():

            anomaly_value = max(
                anomaly_value,
                1e-12
            )

            trust = 1.0 / anomaly_value

            trust = max(
                0.0,
                min(
                    1.0,
                    trust
                )
            )

            current_trust[client_id] = trust

        return current_trust

    def calculate_trust(
            self,
            scores
    ):
        """
        Calculate dynamic trust using:

        Previous trust
        +
        Current round behavior
        """

        relative_anomaly = (
            self.calculate_relative_anomaly(
                scores
            )
        )

        current_trust = (
            self.calculate_current_trust(
                relative_anomaly
            )
        )

        trust_scores = {}

        for client_id in current_trust:

            previous_trust = (
                self.history.get_trust(
                    client_id
                )
            )

            trust_score = (
                    self.history_weight
                    * previous_trust
                    +
                    self.current_weight
                    * current_trust[client_id]
            )

            trust_score = max(
                0.0,
                min(
                    1.0,
                    trust_score
                )
            )

            trust_scores[client_id] = (
                trust_score
            )

        return trust_scores

    def update_history(
            self,
            trust_scores
    ):
        """
        Store the current trust score for
        each client.
        """

        for client_id, trust_score in (
                trust_scores.items()
        ):

            self.history.update_trust(
                client_id,
                trust_score
            )

    def get_trust_zone(
            self,
            trust_score,
            high_threshold=0.75,
            medium_threshold=0.40
    ):
        """
        Assign a trust zone.

        HIGH   >= 0.75
        MEDIUM >= 0.40
        LOW    < 0.40

        These thresholds are configurable and
        are not final research values.
        """

        if trust_score >= high_threshold:

            return "HIGH"

        if trust_score >= medium_threshold:

            return "MEDIUM"

        return "LOW"