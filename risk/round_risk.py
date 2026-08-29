class RoundRisk:

    def __init__(
            self,
            low_risk_threshold=0.30,
            medium_risk_threshold=0.60
    ):

        # Configurable thresholds
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

        # --------------------------------------------------
        # Average anomaly distance
        # --------------------------------------------------

        average_distance = (
                sum(distances.values())
                / len(distances)
        )


        # --------------------------------------------------
        # Normalize anomaly component
        #
        # Higher anomaly = higher risk
        # --------------------------------------------------

        anomaly_component = (
                average_distance
                / (1.0 + average_distance)
        )


        # --------------------------------------------------
        # Trust component
        #
        # Lower average trust = higher risk
        # --------------------------------------------------

        average_trust = (
                sum(trust_scores.values())
                / len(trust_scores)
        )

        trust_component = (
                1.0 - average_trust
        )


        # --------------------------------------------------
        # Suspicious client component
        #
        # Clients below HIGH trust are considered
        # suspicious for the current experimental
        # risk calculation.
        # --------------------------------------------------

        suspicious_clients = sum(
            1
            for trust in trust_scores.values()
            if trust < 0.75
        )


        suspicious_ratio = (
                suspicious_clients
                / len(trust_scores)
        )


        # --------------------------------------------------
        # Combined round risk
        # --------------------------------------------------

        risk_score = (
                0.4 * anomaly_component
                + 0.3 * trust_component
                + 0.3 * suspicious_ratio
        )


        # Keep score between 0 and 1

        risk_score = max(
            0.0,
            min(1.0, risk_score)
        )


        # --------------------------------------------------
        # Determine risk level
        # --------------------------------------------------

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