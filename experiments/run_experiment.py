import csv
import os
import random

import torch

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients
from attacks.label_flip import LabelFlipDataset


# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1

SEED = 42

MALICIOUS_CLIENTS = [9, 10]
FLIP_RATIO = 0.75


# ==========================================================
# RESULT CONFIGURATION
# ==========================================================

RESULT_DIR = "experiments"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "results_dev_adaptive_trust_seed42.csv"
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================================
# CREATE SERVER
# ==========================================================

server = Server()

print()
print("Server created.")
print("Using Original PID Detector.")
print("Using Adaptive Trust Engine.")
print("Using Improved Round Risk.")
print("Using Adaptive Aggregation.")


# ==========================================================
# LOAD MNIST
# ==========================================================

train_dataset, test_dataset = load_mnist()

print("MNIST loaded.")


# ==========================================================
# CREATE CLIENT DATASETS
# ==========================================================

client_datasets = create_clients(
    train_dataset,
    num_clients=NUM_CLIENTS
)


# ==========================================================
# CREATE CLIENTS
# ==========================================================

clients = []

for i in range(NUM_CLIENTS):

    client_id = i + 1

    client_dataset = client_datasets[i]

    # ------------------------------------------------------
    # Apply label-flipping attack
    # ------------------------------------------------------

    if client_id in MALICIOUS_CLIENTS:

        print(
            f"Applying label-flip attack to "
            f"Client {client_id}"
        )

        client_dataset = LabelFlipDataset(
            client_dataset,
            flip_ratio=FLIP_RATIO,
            seed=SEED
        )

    # ------------------------------------------------------
    # Create client
    # ------------------------------------------------------

    client = Client(
        client_id=client_id,
        dataset=client_dataset
    )

    clients.append(client)

    print(
        f"Client {client.client_id} created with "
        f"{len(client.dataset)} samples."
    )


# ==========================================================
# RESULTS STORAGE
# ==========================================================

results = []


# ==========================================================
# FEDERATED TRAINING
# ==========================================================

for round_number in range(
        1,
        NUM_ROUNDS + 1
):

    print()
    print("==============================")
    print(
        f"Federated Round {round_number}"
    )
    print("==============================")


    # ======================================================
    # GET GLOBAL MODEL PARAMETERS
    # ======================================================

    global_parameters = (
        server.global_model.state_dict()
    )


    # ======================================================
    # SEND GLOBAL MODEL TO CLIENTS
    # ======================================================

    for client in clients:

        client.set_model(
            global_parameters
        )


    # ======================================================
    # CONTAINERS
    # ======================================================

    client_parameters = []

    client_sizes = []

    client_updates = {}


    # ======================================================
    # LOCAL TRAINING
    # ======================================================

    for client in clients:

        print(
            f"Training Client "
            f"{client.client_id}"
        )

        client.train(
            epochs=LOCAL_EPOCHS
        )

        parameters = (
            client.get_parameters()
        )

        update = client.get_update(
            global_parameters
        )

        client_parameters.append(
            parameters
        )

        client_sizes.append(
            len(client.dataset)
        )

        client_updates[
            client.client_id
        ] = update


    # ======================================================
    # PID ANOMALY DETECTION
    # ======================================================

    distances, pid_scores = (
        server.detector.calculate_scores(
            client_updates
        )
    )


    # ======================================================
    # ADAPTIVE TRUST ANALYSIS
    # ======================================================

    all_pid_scores = list(
        pid_scores.values()
    )


    # ------------------------------------------------------
    # Calculate relative anomaly
    # ------------------------------------------------------

    relative_anomaly = {}

    for client_id, pid_score in pid_scores.items():

        relative_anomaly[client_id] = (
            server.trust_engine.calculate_relative_anomaly(
                pid_score,
                all_pid_scores
            )
        )


    # ------------------------------------------------------
    # Calculate continuous trust
    # ------------------------------------------------------

    trust_scores = {}

    trust_details = {}

    for client_id, pid_score in pid_scores.items():

        result = (
            server.trust_engine.calculate_trust(
                client_id,
                pid_score,
                all_pid_scores
            )
        )

        trust_scores[client_id] = result["trust"]

        trust_details[client_id] = result


    # ======================================================
    # DISPLAY TRUST ANALYSIS
    # ======================================================

    print()
    print("Trust Analysis")
    print("------------------------------")

    print(
        "Client | PID Score | Relative Anomaly | "
        "Current Trust | Historical Trust | "
        "Persistence | Final Trust"
    )

    print(
        "--------------------------------------------------------------------------"
    )

    for client in clients:

        client_id = client.client_id

        details = trust_details[client_id]

        print(
            f"{client_id:6} | "
            f"{details['pid_score']:9.4f} | "
            f"{details['relative_anomaly']:16.4f} | "
            f"{details['current_trust']:13.4f} | "
            f"{details['historical_trust']:16.4f} | "
            f"{details['persistence']:11.4f} | "
            f"{details['trust']:.4f}"
        )


    # ======================================================
    # ROUND RISK
    # ======================================================

    (
        risk_score,
        risk_level,
        suspicious_clients
    ) = server.round_risk.calculate_risk(
        distances,
        trust_scores
    )


    print()
    print("Round Risk")
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
        f"{suspicious_clients}/"
        f"{NUM_CLIENTS}"
    )


    # ======================================================
    # ADAPTIVE AGGREGATION
    # ======================================================

    (
        new_parameters,
        selected_aggregator
    ) = server.aggregate(
        client_parameters,
        client_sizes,
        trust_scores,
        risk_level,
        client_updates
    )


    # ======================================================
    # CALCULATE EFFECTIVE TRUST WEIGHTS
    # ======================================================

    aggregation_weights = (
        server.adaptive_aggregator.calculate_trust_weights(
            client_sizes,
            trust_scores
        )
    )


    # ======================================================
    # DISPLAY AGGREGATION WEIGHTS
    # ======================================================

    print()
    print("Aggregation Weights")
    print("------------------------------")

    for client_id, weight in enumerate(
            aggregation_weights,
            start=1
    ):

        print(
            f"Client {client_id}: "
            f"{weight:.6f}"
        )


    # ======================================================
    # UPDATE GLOBAL MODEL
    # ======================================================

    server.global_model.load_state_dict(
        new_parameters
    )


    # ======================================================
    # GLOBAL MODEL EVALUATION
    # ======================================================

    accuracy = server.evaluate(
        test_dataset
    )

    accuracy_percent = (
            accuracy * 100
    )


    print()
    print(
        f"Round {round_number} Accuracy: "
        f"{accuracy_percent:.2f}%"
    )


    # ======================================================
    # STORE ROUND RESULTS
    # ======================================================

    row = {

        "seed":
            SEED,

        "round":
            round_number,

        "accuracy":
            accuracy,

        "accuracy_percent":
            accuracy_percent,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "suspicious_clients":
            suspicious_clients,

        "aggregator":
            selected_aggregator
    }


    # ------------------------------------------------------
    # Store PID scores
    # ------------------------------------------------------

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        row[
            f"pid_client_{client_id}"
        ] = pid_scores[client_id]


    # ------------------------------------------------------
    # Store relative anomaly
    # ------------------------------------------------------

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        row[
            f"relative_anomaly_client_{client_id}"
        ] = relative_anomaly[client_id]


    # ------------------------------------------------------
    # Store trust scores
    # ------------------------------------------------------

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        row[
            f"trust_client_{client_id}"
        ] = trust_scores[client_id]


    # ------------------------------------------------------
    # Store aggregation weights
    # ------------------------------------------------------

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        row[
            f"aggregation_weight_client_{client_id}"
        ] = aggregation_weights[
            client_id - 1
            ]


    results.append(row)


# ==========================================================
# SAVE RESULTS
# ==========================================================

fieldnames = results[0].keys()


with open(
        RESULT_FILE,
        "w",
        newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

final_accuracy = (
    results[-1]["accuracy_percent"]
)

best_accuracy = max(
    row["accuracy_percent"]
    for row in results
)


print()
print("==========================================")
print("DEVELOPMENT EXPERIMENT COMPLETED")
print("==========================================")

print()
print("Configuration")
print("------------------------------------------")

print(
    f"Clients: "
    f"{NUM_CLIENTS}"
)

print(
    f"Malicious Clients: "
    f"{MALICIOUS_CLIENTS}"
)

print(
    f"Label Flip Ratio: "
    f"{FLIP_RATIO}"
)

print(
    f"Rounds: "
    f"{NUM_ROUNDS}"
)

print(
    f"Seed: "
    f"{SEED}"
)


print()
print("Results")
print("------------------------------------------")

print(
    f"Final Accuracy: "
    f"{final_accuracy:.2f}%"
)

print(
    f"Best Accuracy: "
    f"{best_accuracy:.2f}%"
)


print()
print(
    f"Results saved to: "
    f"{RESULT_FILE}"
)