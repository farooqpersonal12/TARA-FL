class RoundRisk:

    def __init__(
            self,
            low_risk_threshold=0.30,
            medium_risk_threshold=0.60
    ):

        self.low_risk_threshold = (
            low_risk_threshold
        )

        self.medium_risk_threshold = (
            medium_risk_threshold
        )


    def calculate_risk(
            self,
            distances,
            trust_scores
    ):

        if not distances or not trust_scores:
            return 0.0, "LOW", 0


        # ==================================================
        # 1. AVERAGE ANOMALY COMPONENT
        # ==================================================

        average_distance = (
                sum(distances.values())
                / len(distances)
        )

        anomaly_component = (
                average_distance
                / (1.0 + average_distance)
        )


        # ==================================================
        # 2. AVERAGE TRUST COMPONENT
        # ==================================================

        average_trust = (
                sum(trust_scores.values())
                / len(trust_scores)
        )

        trust_component = (
                1.0 - average_trust
        )


        # ==================================================
        # 3. SUSPICIOUS CLIENT COMPONENT
        # ==================================================

        suspicious_clients = sum(
            1
            for trust in trust_scores.values()
            if trust < 0.75
        )

        suspicious_ratio = (
                suspicious_clients
                / len(trust_scores)
        )


        # ==================================================
        # 4. WORST CLIENT COMPONENT
        # ==================================================
        #
        # New component.
        #
        # Protects against the problem where one extremely
        # malicious client is hidden by many trustworthy
        # clients.
        # ==================================================

        minimum_trust = min(
            trust_scores.values()
        )

        worst_client_component = (
                1.0 - minimum_trust
        )


        # ==================================================
        # 5. COMBINED ROUND RISK
        # ==================================================

        risk_score = (

                0.25 * anomaly_component

                + 0.25 * trust_component

                + 0.20 * suspicious_ratio

                + 0.30 * worst_client_component
        )


        # ==================================================
        # KEEP RISK BETWEEN 0 AND 1
        # ==================================================

        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score
            )
        )


        # ==================================================
        # DETERMINE RISK LEVEL
        # ==================================================

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