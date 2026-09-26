class TrustHistory:
    """
    Stores historical trust and anomaly information
    for each client across federated learning rounds.
    """

    def __init__(self):
        self.trust_history = {}
        self.anomaly_history = {}

    # ---------------------------------------------------------
    # Trust history
    # ---------------------------------------------------------

    def update_trust(self, client_id, trust_score):
        """
        Store a trust score for a client.
        """
        if client_id not in self.trust_history:
            self.trust_history[client_id] = []

        self.trust_history[client_id].append(
            float(trust_score)
        )

    def get_latest_trust(self, client_id):
        """
        Return the most recent trust score.

        New clients have no historical information,
        so None is returned.
        """
        history = self.trust_history.get(client_id, [])

        if not history:
            return None

        return history[-1]

    def get_trust_history(self, client_id):
        """
        Return the complete trust history for a client.
        """
        return self.trust_history.get(
            client_id,
            []
        )

    # ---------------------------------------------------------
    # Anomaly history
    # ---------------------------------------------------------

    def update_anomaly(self, client_id, anomaly_score):
        """
        Store a relative anomaly score for a client.
        """
        if client_id not in self.anomaly_history:
            self.anomaly_history[client_id] = []

        self.anomaly_history[client_id].append(
            float(anomaly_score)
        )

    def get_anomaly_history(self, client_id):
        """
        Return the complete anomaly history for a client.
        """
        return self.anomaly_history.get(
            client_id,
            []
        )

    # ---------------------------------------------------------
    # Combined update
    # ---------------------------------------------------------

    def update(
            self,
            client_id,
            trust_score,
            anomaly_score
    ):
        """
        Store both trust and anomaly information
        for the current round.
        """
        self.update_trust(
            client_id,
            trust_score
        )

        self.update_anomaly(
            client_id,
            anomaly_score
        )