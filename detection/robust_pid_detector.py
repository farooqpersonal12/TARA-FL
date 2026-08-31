import torch


class RobustPIDDetector:

    def __init__(
            self,
            kp=1.0,
            ki=0.08,
            kd=5.0
    ):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.distance_history = {}

    def _flatten_update(self, update):

        tensors = []

        for parameter in update.values():

            if torch.is_floating_point(parameter):

                tensors.append(
                    parameter.detach().float().flatten()
                )

        return torch.cat(tensors)

    def _calculate_distance(
            self,
            update,
            reference
    ):

        update_vector = self._flatten_update(
            update
        )

        distance = torch.norm(
            update_vector - reference,
            p=2
        )

        denominator = torch.norm(
            reference,
            p=2
        )

        normalized_distance = (
                distance / (denominator + 1e-12)
        )

        return normalized_distance.item()

    def calculate_scores(
            self,
            client_updates
    ):

        client_ids = list(
            client_updates.keys()
        )

        update_vectors = []

        for client_id in client_ids:

            vector = self._flatten_update(
                client_updates[client_id]
            )

            update_vectors.append(vector)

        stacked_updates = torch.stack(
            update_vectors
        )

        # --------------------------------------------------
        # Robust reference
        #
        # Use coordinate-wise median instead of mean.
        # --------------------------------------------------

        reference = torch.median(
            stacked_updates,
            dim=0
        ).values

        distances = {}

        for client_id in client_ids:

            distance = self._calculate_distance(
                client_updates[client_id],
                reference
            )

            distances[client_id] = distance

        # --------------------------------------------------
        # PID anomaly scores
        # --------------------------------------------------

        scores = {}

        for client_id in client_ids:

            current_distance = (
                distances[client_id]
            )

            history = self.distance_history.get(
                client_id,
                []
            )

            previous_distance = (
                history[-1]
                if len(history) > 0
                else current_distance
            )

            integral = sum(history)

            derivative = (
                    current_distance
                    - previous_distance
            )

            score = (
                    self.kp * current_distance
                    + self.ki * integral
                    + self.kd * derivative
            )

            score = max(
                0.0,
                score
            )

            scores[client_id] = score

            history.append(
                current_distance
            )

            self.distance_history[
                client_id
            ] = history

        return distances, scores