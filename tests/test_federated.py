import torch

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients
from attacks.label_flip import LabelFlipDataset


def test_federated_pipeline():

    # --------------------------------------------------
    # Create server
    # --------------------------------------------------

    server = Server()

    assert server.global_model is not None
    assert server.detector is not None
    assert server.trust_engine is not None
    assert server.round_risk is not None
    assert server.adaptive_aggregator is not None


    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    train_dataset, test_dataset = load_mnist()

    assert train_dataset is not None
    assert test_dataset is not None


    # --------------------------------------------------
    # Create client datasets
    # --------------------------------------------------

    client_datasets = create_clients(
        train_dataset,
        num_clients=3
    )

    assert len(client_datasets) == 3


    # --------------------------------------------------
    # Apply malicious attack to Client 3
    # --------------------------------------------------

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

    assert len(clients) == 3


    # --------------------------------------------------
    # Get initial global model
    # --------------------------------------------------

    global_parameters = (
        server.global_model.state_dict()
    )


    # --------------------------------------------------
    # Local training
    # --------------------------------------------------

    client_parameters = []
    client_updates = {}
    client_sizes = []

    for client in clients:

        client.set_model(
            global_parameters
        )

        client.train(
            epochs=1
        )

        parameters = client.get_parameters()

        update = client.get_update(
            global_parameters
        )

        client_parameters.append(
            parameters
        )

        client_updates[
            client.client_id
        ] = update

        client_sizes.append(
            len(client.dataset)
        )


    # --------------------------------------------------
    # Verify client updates
    # --------------------------------------------------

    assert len(client_parameters) == 3
    assert len(client_updates) == 3
    assert len(client_sizes) == 3


    # --------------------------------------------------
    # PID anomaly detection
    # --------------------------------------------------

    distances, pid_scores = (
        server.detector.calculate_scores(
            client_updates
        )
    )

    assert len(distances) == 3
    assert len(pid_scores) == 3


    # --------------------------------------------------
    # Calculate trust
    # --------------------------------------------------

    trust_scores = (
        server.trust_engine.calculate_trust(
            pid_scores
        )
    )

    assert len(trust_scores) == 3

    for client_id in trust_scores:

        assert 0.0 <= trust_scores[client_id] <= 1.0


    # --------------------------------------------------
    # Update trust history
    # --------------------------------------------------

    server.trust_engine.update_history(
        trust_scores
    )


    # --------------------------------------------------
    # Calculate round risk
    # --------------------------------------------------

    risk_score, risk_level, suspicious_clients = (
        server.round_risk.calculate_risk(
            distances,
            trust_scores
        )
    )

    assert 0.0 <= risk_score <= 1.0

    assert risk_level in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    assert 0 <= suspicious_clients <= 3


    # --------------------------------------------------
    # Adaptive aggregation
    # --------------------------------------------------

    new_parameters, selected_aggregator = (
        server.aggregate(
            client_parameters,
            client_sizes,
            trust_scores,
            risk_level
        )
    )


    # --------------------------------------------------
    # Verify aggregation result
    # --------------------------------------------------

    assert new_parameters is not None

    assert selected_aggregator in [
        "TRUST_AWARE_FEDAVG",
        "TRIMMED_MEAN",
        "MEDIAN"
    ]


    # --------------------------------------------------
    # Update global model
    # --------------------------------------------------

    server.global_model.load_state_dict(
        new_parameters
    )


    # --------------------------------------------------
    # Evaluate global model
    # --------------------------------------------------

    accuracy = server.evaluate(
        test_dataset
    )

    assert 0.0 <= accuracy <= 1.0