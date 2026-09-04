import torch


class AdaptiveAggregator:

    def __init__(
            self,
            low_threshold=0.30,
            medium_threshold=0.60,
            trim_ratio=0.25,
            trust_floor=0.05
    ):

        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.trim_ratio = trim_ratio
        self.trust_floor = trust_floor

    # ======================================================
    # SELECT AGGREGATION STRATEGY
    # ======================================================

    def select_aggregator(self, risk_level):

        if risk_level == "LOW":

            return "TRUST_AWARE_FEDAVG"

        elif risk_level == "MEDIUM":

            return "TRUST_WEIGHTED_ROBUST"

        elif risk_level == "HIGH":

            return "TRUST_WEIGHTED_MEDIAN"

        raise ValueError(
            f"Unknown risk level: {risk_level}"
        )

    # ======================================================
    # CALCULATE TRUST WEIGHTS
    # ======================================================

    def calculate_trust_weights(
            self,
            client_sizes,
            trust_scores
    ):
        """
        Calculate aggregation influence using:

            client_size × trust

        A small trust floor prevents numerical instability
        while still allowing very low-trust clients to have
        extremely small influence.
        """

        weights = []

        for client_id, size in enumerate(
                client_sizes,
                start=1
        ):

            trust = trust_scores.get(
                client_id,
                0.0
            )

            trust = max(
                0.0,
                min(
                    1.0,
                    trust
                )
            )

            # --------------------------------------------------
            # Trust floor
            # --------------------------------------------------

            if trust > 0.0:

                effective_trust = max(
                    trust,
                    self.trust_floor
                )

            else:

                effective_trust = 0.0

            weights.append(
                size * effective_trust
            )

        total_weight = sum(weights)

        # --------------------------------------------------
        # Safety fallback
        # --------------------------------------------------

        if total_weight <= 1e-12:

            total_size = sum(
                client_sizes
            )

            return [
                size / total_size
                for size in client_sizes
            ]

        return [
            weight / total_weight
            for weight in weights
        ]

    # ======================================================
    # TRUST-AWARE FEDAVG
    # ======================================================

    def trust_aware_fedavg(
            self,
            client_parameters,
            client_sizes,
            trust_scores
    ):
        """
        Standard FedAvg modified by client trust.

        Effective weight:

            client_size × trust
        """

        normalized_weights = (
            self.calculate_trust_weights(
                client_sizes,
                trust_scores
            )
        )

        new_parameters = {}

        for name in client_parameters[0]:

            new_parameters[name] = (
                torch.zeros_like(
                    client_parameters[0][name]
                )
            )

            for parameters, weight in zip(
                    client_parameters,
                    normalized_weights
            ):

                new_parameters[name] += (
                        weight *
                        parameters[name]
                )

        return new_parameters

    # ======================================================
    # UPDATE DISTANCE
    # ======================================================

    def calculate_update_distances(
            self,
            client_updates
    ):
        """
        Calculate each client's distance from the
        update centroid.

        This is used for robust client filtering.

        The client identity is preserved, unlike
        coordinate-wise trimmed mean.
        """

        if not client_updates:

            return {}

        client_ids = list(
            client_updates.keys()
        )

        vectors = []

        for client_id in client_ids:

            tensors = []

            for parameter in (
                    client_updates[client_id].values()
            ):

                if torch.is_floating_point(
                        parameter
                ):

                    tensors.append(
                        parameter.detach()
                        .float()
                        .flatten()
                    )

            vector = torch.cat(
                tensors
            )

            vectors.append(vector)

        stacked = torch.stack(
            vectors
        )

        centroid = torch.mean(
            stacked,
            dim=0
        )

        distances = {}

        for client_id, vector in zip(
                client_ids,
                vectors
        ):

            distance = torch.norm(
                vector - centroid,
                p=2
            )

            distances[client_id] = (
                distance.item()
            )

        return distances

    # ======================================================
    # ROBUST CLIENT SELECTION
    # ======================================================

    def select_robust_clients(
            self,
            client_ids,
            distances,
            risk_level
    ):
        """
        Remove the most extreme client updates.

        LOW:
            No filtering.

        MEDIUM:
            Remove approximately trim_ratio
            of the most anomalous clients.

        HIGH:
            Remove a stronger fraction of anomalous
            clients.
        """

        if risk_level == "LOW":

            return list(client_ids)

        if not distances:

            return list(client_ids)

        if risk_level == "MEDIUM":

            trim_ratio = self.trim_ratio

        else:

            # Stronger filtering under HIGH risk
            trim_ratio = min(
                0.40,
                self.trim_ratio * 1.5
            )

        num_clients = len(
            client_ids
        )

        remove_count = int(
            num_clients * trim_ratio
        )

        # Never remove every client.
        remove_count = min(
            remove_count,
            max(0, num_clients - 1)
        )

        if remove_count == 0:

            return list(client_ids)

        sorted_clients = sorted(
            client_ids,
            key=lambda client_id:
            distances[client_id]
        )

        selected_clients = (
            sorted_clients[
                :num_clients - remove_count
            ]
        )

        return selected_clients

    # ======================================================
    # TRUST-WEIGHTED ROBUST AGGREGATION
    # ======================================================

    def trust_weighted_robust(
            self,
            client_parameters,
            client_sizes,
            trust_scores,
            client_updates,
            risk_level
    ):
        """
        Trust-aware robust aggregation.

        Process:

            1. Calculate update distances.
            2. Identify extreme clients.
            3. Remove extreme clients according to
               the round risk.
            4. Weight remaining clients using:

                   client_size × trust

        This preserves the identity of every client
        throughout the aggregation process.
        """

        client_ids = list(
            range(
                1,
                len(client_parameters) + 1
            )
        )

        distances = (
            self.calculate_update_distances(
                client_updates
            )
        )

        selected_clients = (
            self.select_robust_clients(
                client_ids,
                distances,
                risk_level
            )
        )

        # --------------------------------------------------
        # Calculate weights only for selected clients
        # --------------------------------------------------

        selected_weights = []

        for client_id in selected_clients:

            index = client_id - 1

            trust = trust_scores.get(
                client_id,
                0.0
            )

            trust = max(
                0.0,
                min(
                    1.0,
                    trust
                )
            )

            if trust > 0.0:

                effective_trust = max(
                    trust,
                    self.trust_floor
                )

            else:

                effective_trust = 0.0

            weight = (
                    client_sizes[index]
                    * effective_trust
            )

            selected_weights.append(
                weight
            )

        total_weight = sum(
            selected_weights
        )

        # --------------------------------------------------
        # Safety fallback
        # --------------------------------------------------

        if total_weight <= 1e-12:

            selected_weights = [
                client_sizes[client_id - 1]
                for client_id in selected_clients
            ]

            total_weight = sum(
                selected_weights
            )

        normalized_weights = [
            weight / total_weight
            for weight in selected_weights
        ]

        # --------------------------------------------------
        # Aggregate
        # --------------------------------------------------

        new_parameters = {}

        for name in client_parameters[0]:

            new_parameters[name] = (
                torch.zeros_like(
                    client_parameters[0][name]
                )
            )

            for client_id, weight in zip(
                    selected_clients,
                    normalized_weights
            ):

                index = client_id - 1

                new_parameters[name] += (
                        weight *
                        client_parameters[index][name]
                )

        return (
            new_parameters,
            selected_clients,
            distances
        )

    # ======================================================
    # TRUST-WEIGHTED MEDIAN
    # ======================================================

    def trust_weighted_median(
            self,
            client_parameters,
            client_sizes,
            trust_scores
    ):
        """
        Coordinate-wise weighted median.

        Trust is converted into aggregation weight.
        The weighted median is determined using cumulative
        trust-weighted contribution.
        """

        normalized_weights = (
            self.calculate_trust_weights(
                client_sizes,
                trust_scores
            )
        )

        new_parameters = {}

        for name in client_parameters[0]:

            stacked = torch.stack([
                parameters[name]
                for parameters in client_parameters
            ])

            original_shape = stacked.shape[1:]

            flattened = stacked.reshape(
                stacked.shape[0],
                -1
            )

            result = torch.zeros(
                flattened.shape[1],
                dtype=stacked.dtype,
                device=stacked.device
            )

            for coordinate in range(
                    flattened.shape[1]
            ):

                values = flattened[
                    :,
                    coordinate
                ]

                sorted_indices = torch.argsort(
                    values
                )

                cumulative_weight = 0.0

                selected_index = (
                    sorted_indices[-1].item()
                )

                for index in sorted_indices:

                    client_index = (
                        index.item()
                    )

                    cumulative_weight += (
                        normalized_weights[
                            client_index
                        ]
                    )

                    if cumulative_weight >= 0.5:

                        selected_index = (
                            client_index
                        )

                        break

                result[coordinate] = (
                    values[selected_index]
                )

            new_parameters[name] = (
                result.reshape(
                    original_shape
                )
            )

        return new_parameters

    # ======================================================
    # ADAPTIVE AGGREGATION
    # ======================================================

    def aggregate(
            self,
            client_parameters,
            client_sizes,
            trust_scores,
            risk_level,
            client_updates=None
    ):
        """
        Select and execute aggregation according
        to round risk.

        Trust is incorporated at every risk level.
        """

        aggregator = (
            self.select_aggregator(
                risk_level
            )
        )

        # --------------------------------------------------
        # LOW RISK
        # --------------------------------------------------

        if aggregator == "TRUST_AWARE_FEDAVG":

            new_parameters = (
                self.trust_aware_fedavg(
                    client_parameters,
                    client_sizes,
                    trust_scores
                )
            )

            selected_clients = list(
                range(
                    1,
                    len(client_parameters) + 1
                )
            )

        # --------------------------------------------------
        # MEDIUM RISK
        # --------------------------------------------------

        elif aggregator == "TRUST_WEIGHTED_ROBUST":

            if client_updates is None:

                raise ValueError(
                    "client_updates are required "
                    "for TRUST_WEIGHTED_ROBUST"
                )

            (
                new_parameters,
                selected_clients,
                distances
            ) = self.trust_weighted_robust(
                client_parameters,
                client_sizes,
                trust_scores,
                client_updates,
                risk_level
            )

        # --------------------------------------------------
        # HIGH RISK
        # --------------------------------------------------

        elif aggregator == "TRUST_WEIGHTED_MEDIAN":

            new_parameters = (
                self.trust_weighted_median(
                    client_parameters,
                    client_sizes,
                    trust_scores
                )
            )

            selected_clients = list(
                range(
                    1,
                    len(client_parameters) + 1
                )
            )

        else:

            raise ValueError(
                f"Unsupported aggregator: "
                f"{aggregator}"
            )

        return (
            new_parameters,
            aggregator
        )