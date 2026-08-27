class TrustHistory:

    def __init__(self, initial_trust=1.0):

        self.initial_trust = initial_trust

        self.history = {}

    def get_trust(self, client_id):

        client_history = self.history.get(
            client_id,
            []
        )

        if not client_history:
            return self.initial_trust

        return client_history[-1]

    def update_trust(
            self,
            client_id,
            trust_score
    ):

        self.history.setdefault(
            client_id,
            []
        )

        self.history[client_id].append(
            float(trust_score)
        )

    def get_history(self, client_id):

        return self.history.get(
            client_id,
            []
        )