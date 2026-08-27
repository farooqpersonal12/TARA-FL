from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients

from attacks.label_flip import LabelFlipDataset
from detection.pid_detector import PIDDetector
from trustengine.trust_engine import TrustEngine


if __name__ == "__main__":

    # --------------------------------------------------
    # Server
    # --------------------------------------------------

    server = Server()

    print("Server created.")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset, test_dataset = load_mnist()

    print("MNIST loaded.")

    # --------------------------------------------------
    # Create client datasets
    # --------------------------------------------------

    client_datasets = create_clients(
        train_dataset,
        num_clients=3
    )

    # Client 3 is malicious
    client_datasets[2] = LabelFlipDataset(
        client_datasets[2],
        flip_ratio=0.75,
        seed=42
    )

    # --------------------------------------------------
    # Create clients
    # --------------------------------------------------

    clients = []

    for i in range(3):

        client = Client(
            client_id=i + 1,
            dataset=client_datasets[i]
        )

        clients.append(client)

        print(
            "Client",
            client.client_id,
            "created with",
            len(client.dataset),
            "samples."
        )

    # --------------------------------------------------
    # Detection and Trust Engine
    # --------------------------------------------------

    detector = PIDDetector(
        kp=1.0,
        ki=0.08,
        kd=5.0
    )

    trust_engine = TrustEngine(
        history_weight=0.7,
        current_weight=0.3
    )

    # --------------------------------------------------
    # Federated Training
    # --------------------------------------------------

    num_rounds = 5

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
        # Get current global model
        # --------------------------------------------------

        global_parameters = (
            server.global_model.state_dict()
        )

        # --------------------------------------------------
        # Send global model to clients
        # --------------------------------------------------

        for client in clients:

            client.set_model(
                global_parameters
            )

        # --------------------------------------------------
        # Prepare client results
        # --------------------------------------------------

        client_parameters = []
        client_updates = {}
        client_sizes = []

        # --------------------------------------------------
        # Local Training
        # --------------------------------------------------

        for client in clients:

            print(
                "Training Client",
                client.client_id
            )

            client.train(
                epochs=1
            )

            # Local model parameters
            parameters = client.get_parameters()

            # Local model update
            update = client.get_update(
                global_parameters
            )

            # Store parameters
            client_parameters.append(
                parameters
            )

            # Store update
            client_updates[
                client.client_id
            ] = update

            # Store dataset size
            client_sizes.append(
                len(client.dataset)
            )

        # --------------------------------------------------
        # PID-Based Client Behavior Analysis
        # --------------------------------------------------

        distances, scores = (
            detector.calculate_scores(
                client_updates
            )
        )

        print()
        print("Client Behavior Scores")
        print("----------------------")

        for client_id in scores:

            print(
                "Client",
                client_id,
                "distance:",
                round(
                    distances[client_id],
                    6
                ),
                "PID score:",
                round(
                    scores[client_id],
                    6
                )
            )

        # --------------------------------------------------
        # TARA-FL Trust Engine
        # --------------------------------------------------

        relative_anomaly = (
            trust_engine.calculate_relative_anomaly(
                scores
            )
        )

        current_trust = (
            trust_engine.calculate_current_trust(
                relative_anomaly
            )
        )

        trust_scores = (
            trust_engine.calculate_trust(
                scores
            )
        )

        # Save trust history
        trust_engine.update_history(
            trust_scores
        )

        # --------------------------------------------------
        # Relative Anomaly Output
        # --------------------------------------------------

        print()
        print("Relative Anomaly")
        print("----------------")

        for client_id in relative_anomaly:

            print(
                "Client",
                client_id,
                "relative anomaly:",
                round(
                    relative_anomaly[client_id],
                    4
                )
            )

        # --------------------------------------------------
        # Current Trust Evidence Output
        # --------------------------------------------------

        print()
        print("Current Trust Evidence")
        print("----------------------")

        for client_id in current_trust:

            print(
                "Client",
                client_id,
                "current trust:",
                round(
                    current_trust[client_id],
                    4
                )
            )

        # --------------------------------------------------
        # Final Trust Score Output
        # --------------------------------------------------

        print()
        print("Client Trust Scores")
        print("-------------------")

        for client_id in trust_scores:

            trust_score = (
                trust_scores[client_id]
            )

            trust_zone = (
                trust_engine.get_trust_zone(
                    trust_score
                )
            )

            print(
                "Client",
                client_id,
                "trust:",
                round(
                    trust_score,
                    4
                ),
                "zone:",
                trust_zone
            )

        # --------------------------------------------------
        # Standard FedAvg
        # --------------------------------------------------

        print()
        print("Performing FedAvg...")

        new_parameters = (
            server.aggregate(
                client_parameters,
                client_sizes
            )
        )

        server.global_model.load_state_dict(
            new_parameters
        )

        # --------------------------------------------------
        # Global Model Evaluation
        # --------------------------------------------------

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

    # --------------------------------------------------
    # Training Completed
    # --------------------------------------------------

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