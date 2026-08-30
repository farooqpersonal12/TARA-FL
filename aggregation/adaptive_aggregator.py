import torch


class AdaptiveAggregator:

    def __init__(
            self,
            low_threshold=0.30,
            medium_threshold=0.60,
            trim_ratio=0.25
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.trim_ratio = trim_ratio

    # --------------------------------------------------
    # Decide aggregation method from round risk
    # --------------------------------------------------

    def select_aggregator(self, risk_level):

        if risk_level == "LOW":
            return "TRUST_AWARE_FEDAVG"

        elif risk_level == "MEDIUM":
            return "TRIMMED_MEAN"

        elif risk_level == "HIGH":
            return "MEDIAN"

        raise ValueError(
            f"Unknown risk level: {risk_level}"
        )

    # --------------------------------------------------
    # Trust-aware FedAvg
    # --------------------------------------------------

    def trust_aware_fedavg(
            self,
            client_parameters,
            client_sizes,
            trust_scores
    ):

        adjusted_weights = []

        for client_id, size in enumerate(
                client_sizes,
                start=1
        ):

            trust = trust_scores[client_id]

            adjusted_weights.append(
                size * trust
            )

        total_weight = sum(adjusted_weights)

        # Safety fallback
        if total_weight == 0:
            total_weight = sum(client_sizes)

            normalized_weights = [
                size / total_weight
                for size in client_sizes
            ]

        else:
            normalized_weights = [
                weight / total_weight
                for weight in adjusted_weights
            ]

        new_parameters = {}

        for name in client_parameters[0]:

            new_parameters[name] = torch.zeros_like(
                client_parameters[0][name]
            )

            for parameters, weight in zip(
                    client_parameters,
                    normalized_weights
            ):

                new_parameters[name] += (
                        weight * parameters[name]
                )

        return new_parameters

    # --------------------------------------------------
    # Coordinate-wise Median
    # --------------------------------------------------

    def median(
            self,
            client_parameters
    ):

        new_parameters = {}

        for name in client_parameters[0]:

            stacked = torch.stack([
                parameters[name]
                for parameters in client_parameters
            ])

            new_parameters[name] = torch.median(
                stacked,
                dim=0
            ).values

        return new_parameters

    # --------------------------------------------------
    # Coordinate-wise Trimmed Mean
    # --------------------------------------------------

    def trimmed_mean(
            self,
            client_parameters
    ):

        new_parameters = {}

        for name in client_parameters[0]:

            stacked = torch.stack([
                parameters[name]
                for parameters in client_parameters
            ])

            num_clients = stacked.shape[0]

            trim_count = int(
                num_clients * self.trim_ratio
            )

            # Prevent removing every value
            if trim_count * 2 >= num_clients:
                trim_count = max(
                    0,
                    (num_clients - 1) // 2
                )

            sorted_values = torch.sort(
                stacked,
                dim=0
            ).values

            if trim_count > 0:

                sorted_values = sorted_values[
                    trim_count:
                    num_clients - trim_count
                ]

            new_parameters[name] = (
                sorted_values.mean(dim=0)
            )

        return new_parameters

    # --------------------------------------------------
    # Adaptive Aggregation
    # --------------------------------------------------

    def aggregate(
            self,
            client_parameters,
            client_sizes,
            trust_scores,
            risk_level
    ):

        aggregator = self.select_aggregator(
            risk_level
        )

        if aggregator == "TRUST_AWARE_FEDAVG":

            new_parameters = self.trust_aware_fedavg(
                client_parameters,
                client_sizes,
                trust_scores
            )

        elif aggregator == "TRIMMED_MEAN":

            new_parameters = self.trimmed_mean(
                client_parameters
            )

        elif aggregator == "MEDIAN":

            new_parameters = self.median(
                client_parameters
            )

        else:
            raise ValueError(
                f"Unsupported aggregator: {aggregator}"
            )

        return new_parameters, aggregator