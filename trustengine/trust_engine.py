from trustengine.trust_history import TrustHistory


class TrustEngine:

    def __init__(
            self,
            history_weight=0.6,
            current_weight=0.4,
            persistence_penalty=0.15,
            abnormality_threshold=1.25,
            decay_factor=0.8
    ):

        # --------------------------------------------------
        # Trust configuration
        # --------------------------------------------------

        self.history_weight = history_weight
        self.current_weight = current_weight

        self.persistence_penalty = (
            persistence_penalty
        )

        self.abnormality_threshold = (
            abnormality_threshold
        )

        # --------------------------------------------------
        # Recency configuration
        # --------------------------------------------------

        self.decay_factor = decay_factor

        # --------------------------------------------------
        # Historical trust
        # --------------------------------------------------

        self.history = TrustHistory(
            initial_trust=1.0
        )

        # --------------------------------------------------
        # Historical relative anomaly
        # --------------------------------------------------

        self.anomaly_history = {}


    # ======================================================
    # RELATIVE ANOMALY
    # ======================================================

    def calculate_relative_anomaly(
            self,
            scores
    ):

        if not scores:
            return {}

        values = sorted(
            scores.values()
        )

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


    # ======================================================
    # CURRENT TRUST
    # ======================================================

    def calculate_current_trust(
            self,
            relative_anomaly
    ):

        current_trust = {}

        for client_id, anomaly in (
                relative_anomaly.items()
        ):

            anomaly = max(
                anomaly,
                1e-12
            )

            trust = 1.0 / anomaly

            trust = max(
                0.0,
                min(
                    1.0,
                    trust
                )
            )

            current_trust[client_id] = trust

        return current_trust


    # ======================================================
    # RECENCY-WEIGHTED PERSISTENCE
    # ======================================================

    def calculate_persistence(
            self,
            relative_anomaly
    ):
        """
        Calculate persistence of abnormal behavior.

        Recent rounds receive higher importance than
        older rounds.

        decay_factor = 0.8 means:

            newest round      -> highest weight
            previous round    -> 0.8
            older round       -> 0.8^2
            older round       -> 0.8^3
            ...

        Only behavior above abnormality_threshold is
        considered persistent abnormal behavior.
        """

        persistence = {}

        for client_id in relative_anomaly:

            history = self.anomaly_history.get(
                client_id,
                []
            )

            if not history:

                persistence[client_id] = 0.0
                continue

            weighted_abnormal = 0.0
            total_weight = 0.0

            # --------------------------------------------------
            # Newest historical observation receives
            # the highest weight.
            # --------------------------------------------------

            reversed_history = list(
                reversed(history)
            )

            for index, anomaly in enumerate(
                    reversed_history
            ):

                weight = (
                        self.decay_factor ** index
                )

                total_weight += weight

                if anomaly >= (
                        self.abnormality_threshold
                ):

                    weighted_abnormal += weight

            if total_weight == 0:

                persistence_score = 0.0

            else:

                persistence_score = (
                        weighted_abnormal
                        / total_weight
                )

            persistence[client_id] = (
                persistence_score
            )

        return persistence


    # ======================================================
    # DYNAMIC TRUST
    # ======================================================

    def calculate_trust(
            self,
            scores
    ):

        if not scores:
            return {}

        # --------------------------------------------------
        # Calculate peer-relative anomaly
        # --------------------------------------------------

        relative_anomaly = (
            self.calculate_relative_anomaly(
                scores
            )
        )

        # --------------------------------------------------
        # Calculate current-round trust
        # --------------------------------------------------

        current_trust = (
            self.calculate_current_trust(
                relative_anomaly
            )
        )

        # --------------------------------------------------
        # Calculate persistent behavior
        # --------------------------------------------------

        persistence = (
            self.calculate_persistence(
                relative_anomaly
            )
        )

        trust_scores = {}

        for client_id in current_trust:

            # --------------------------------------------------
            # Historical trust
            # --------------------------------------------------

            previous_trust = (
                self.history.get_trust(
                    client_id
                )
            )

            # --------------------------------------------------
            # Current trust
            # --------------------------------------------------

            current = current_trust[
                client_id
            ]

            # --------------------------------------------------
            # Persistent anomaly
            # --------------------------------------------------

            persistent_behavior = (
                persistence[client_id]
            )

            # --------------------------------------------------
            # Historical + current trust
            # --------------------------------------------------

            trust_score = (
                    self.history_weight
                    * previous_trust
                    +
                    self.current_weight
                    * current
            )

            # --------------------------------------------------
            # Persistent anomaly penalty
            # --------------------------------------------------

            penalty = (
                    self.persistence_penalty
                    * persistent_behavior
            )

            trust_score -= penalty

            # --------------------------------------------------
            # Keep trust in [0, 1]
            # --------------------------------------------------

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


    # ======================================================
    # UPDATE HISTORY
    # ======================================================

    def update_history(
            self,
            trust_scores,
            relative_anomaly=None
    ):
        """
        Store trust history and anomaly history.
        """

        # --------------------------------------------------
        # Trust history
        # --------------------------------------------------

        for client_id, trust_score in (
                trust_scores.items()
        ):

            self.history.update_trust(
                client_id,
                trust_score
            )

        # --------------------------------------------------
        # Anomaly history
        # --------------------------------------------------

        if relative_anomaly is not None:

            for client_id, anomaly in (
                    relative_anomaly.items()
            ):

                if client_id not in (
                        self.anomaly_history
                ):

                    self.anomaly_history[
                        client_id
                    ] = []

                self.anomaly_history[
                    client_id
                ].append(anomaly)


    # ======================================================
    # TRUST ZONE
    # ======================================================

    def get_trust_zone(
            self,
            trust_score,
            high_threshold=0.75,
            medium_threshold=0.40
    ):

        if trust_score >= high_threshold:

            return "HIGH"

        if trust_score >= medium_threshold:

            return "MEDIUM"

        return "LOW"