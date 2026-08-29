from model.model import MNISTModel
from clients.client import Client
from data.dataset import load_mnist, create_clients

import torch

from detection.pid_detector import PIDDetector
from trustengine.trust_engine import TrustEngine
from risk.round_risk import RoundRisk
from attacks.label_flip import LabelFlipDataset


class Server:

    def __init__(self):

        self.global_model = MNISTModel()

        # --------------------------------------------------
        # Malicious / anomaly detection
        # --------------------------------------------------

        self.detector = PIDDetector()

        # --------------------------------------------------
        # Dynamic client trust
        # --------------------------------------------------

        self.trust_engine = TrustEngine()

        # --------------------------------------------------
        # Round risk assessment
        # --------------------------------------------------

        self.round_risk = RoundRisk()


    def aggregate(
            self,
            client_parameters,
            client_sizes,
            trust_scores
    ):

        # --------------------------------------------------
        # Calculate trust-adjusted weights
        # --------------------------------------------------

        adjusted_weights = []

        for client_id, size in enumerate(
                client_sizes,
                start=1
        ):

            trust = trust_scores[client_id]

            # Sample size × trust score
            adjusted_weight = (
                    size * trust
            )

            adjusted_weights.append(
                adjusted_weight
            )


        # --------------------------------------------------
        # Normalize weights
        # --------------------------------------------------

        total_weight = sum(
            adjusted_weights
        )

        normalized_weights = [
            weight / total_weight
            for weight in adjusted_weights
        ]


        # --------------------------------------------------
        # Display Trust-Aware Aggregation Weights
        # --------------------------------------------------

        print()
        print("Trust-Aware Aggregation")
        print("------------------------------")

        for client_id, weight in enumerate(
                normalized_weights,
                start=1
        ):

            print(
                f"Client {client_id} "
                f"aggregation weight: "
                f"{weight:.4f}"
            )


        # --------------------------------------------------
        # Aggregate Client Models
        # --------------------------------------------------

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
                        weight * parameters[name]
                )


        return new_parameters


    def evaluate(
            self,
            test_dataset
    ):

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )

        self.global_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                outputs = self.global_model(
                    images
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                total += labels.size(0)

                correct += (
                        predictions == labels
                ).sum().item()


        accuracy = correct / total

        return accuracy


# ==========================================================
# MAIN FEDERATED LEARNING PIPELINE
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Create Server
    # ------------------------------------------------------

    server = Server()

    print("Server created.")


    # ------------------------------------------------------
    # Load MNIST
    # ------------------------------------------------------

    train_dataset, test_dataset = load_mnist()

    print("MNIST loaded.")


    # ------------------------------------------------------
    # Create Client Datasets
    # ------------------------------------------------------

    client_datasets = create_clients(
        train_dataset,
        num_clients=3
    )


    # ------------------------------------------------------
    # Create Clients
    # ------------------------------------------------------

    clients = []

    for i in range(3):

        client_dataset = client_datasets[i]


        # --------------------------------------------------
        # Client 3 = Malicious
        # --------------------------------------------------

        if i == 2:

            print(
                "Applying label-flip attack to Client 3"
            )

            client_dataset = LabelFlipDataset(
                dataset=client_dataset,
                flip_ratio=0.5,
                seed=42
            )


        client = Client(
            client_id=i + 1,
            dataset=client_dataset
        )

        clients.append(client)

        print(
            "Client",
            client.client_id,
            "created with",
            len(client.dataset),
            "samples."
        )


    # ------------------------------------------------------
    # Federated Training Configuration
    # ------------------------------------------------------

    num_rounds = 5


    # ======================================================
    # FEDERATED TRAINING
    # ======================================================

    for round_number in range(
            1,
            num_rounds + 1
    ):

        print()
        print("==============================")
        print(
            "Federated Round",
            round_number
        )
        print("==============================")


        # --------------------------------------------------
        # Get Current Global Model
        # --------------------------------------------------

        global_parameters = (
            server.global_model.state_dict()
        )


        # --------------------------------------------------
        # Send Global Model to Every Client
        # --------------------------------------------------

        for client in clients:

            client.set_model(
                global_parameters
            )


        # --------------------------------------------------
        # Containers for Client Results
        # --------------------------------------------------

        client_parameters = []

        client_sizes = []

        client_updates = {}


        # ==================================================
        # LOCAL CLIENT TRAINING
        # ==================================================

        for client in clients:

            print(
                "Training Client",
                client.client_id
            )


            # ----------------------------------------------
            # Local Training
            # ----------------------------------------------

            client.train(
                epochs=1
            )


            # ----------------------------------------------
            # Get Trained Parameters
            # ----------------------------------------------

            parameters = (
                client.get_parameters()
            )


            # ----------------------------------------------
            # Calculate Client Update
            # ----------------------------------------------

            update = client.get_update(
                global_parameters
            )


            # ----------------------------------------------
            # Store Parameters
            # ----------------------------------------------

            client_parameters.append(
                parameters
            )


            # ----------------------------------------------
            # Store Dataset Size
            # ----------------------------------------------

            client_sizes.append(
                len(client.dataset)
            )


            # ----------------------------------------------
            # Store Client Update
            # ----------------------------------------------

            client_updates[
                client.client_id
            ] = update


        # ==================================================
        # TRUST ANALYSIS
        # ==================================================

        print()
        print("Trust Analysis")
        print("------------------------------")


        # --------------------------------------------------
        # PID Anomaly Detection
        # --------------------------------------------------

        distances, pid_scores = (
            server.detector.calculate_scores(
                client_updates
            )
        )


        # --------------------------------------------------
        # Calculate Dynamic Trust
        # --------------------------------------------------

        trust_scores = (
            server.trust_engine.calculate_trust(
                pid_scores
            )
        )


        # --------------------------------------------------
        # Update Trust History
        # --------------------------------------------------

        server.trust_engine.update_history(
            trust_scores
        )


        # --------------------------------------------------
        # Display Trust Information
        # --------------------------------------------------

        for client in clients:

            client_id = client.client_id

            trust_score = (
                trust_scores[client_id]
            )

            zone = (
                server.trust_engine.get_trust_zone(
                    trust_score
                )
            )


            print(
                f"Client {client_id}"
            )

            print(
                f"  Anomaly Distance: "
                f"{distances[client_id]:.4f}"
            )

            print(
                f"  PID Score: "
                f"{pid_scores[client_id]:.4f}"
            )

            print(
                f"  Trust Score: "
                f"{trust_score:.4f}"
            )

            print(
                f"  Trust Zone: "
                f"{zone}"
            )


        # ==================================================
        # ROUND RISK ASSESSMENT
        # ==================================================

        risk_score, risk_level, suspicious_clients = (
            server.round_risk.calculate_risk(
                distances,
                trust_scores
            )
        )


        print()
        print("Round Risk Assessment")
        print("------------------------------")

        print(
            f"Risk Score: "
            f"{risk_score:.4f}"
        )

        print(
            f"Risk Level: "
            f"{risk_level}"
        )

        print(
            f"Suspicious Clients: "
            f"{suspicious_clients}/{len(clients)}"
        )


        # ==================================================
        # TRUST-AWARE AGGREGATION
        #
        # Round risk does NOT change the aggregator yet.
        # ==================================================

        new_parameters = server.aggregate(
            client_parameters,
            client_sizes,
            trust_scores
        )


        # --------------------------------------------------
        # Update Global Model
        # --------------------------------------------------

        server.global_model.load_state_dict(
            new_parameters
        )


        # ==================================================
        # GLOBAL MODEL EVALUATION
        # ==================================================

        accuracy = server.evaluate(
            test_dataset
        )


        print(
            "Round",
            round_number,
            "accuracy:",
            accuracy * 100,
            "%"
        )


    # ======================================================
    # TRAINING COMPLETED
    # ======================================================

    print()
    print("==============================")
    print("Federated training completed.")
    print("==============================")


    print(
        "Number of clients:",
        len(clients)
    )


    print(
        "Client sample sizes:",
        client_sizes
    )