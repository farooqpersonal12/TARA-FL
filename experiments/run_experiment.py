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

# ----------------------------------------------------------
# EXP-11
#
# 10 clients
# 1 malicious client
# 50% label flipping
# Original PID Detector
# Improved Trust Engine
# Improved Round Risk
# Adaptive Aggregation
# ----------------------------------------------------------

MALICIOUS_CLIENTS = [9,10]
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
    "results_exp15_improved_risk-v2_10clients_2malicious_75pct.csv"
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

print("Server created.")
print("Using Original PID Detector.")
print("Using Improved Trust Engine.")
print("Using Improved Round Risk.")


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
    # only to malicious clients
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

        # --------------------------------------------------
        # Train local model
        # --------------------------------------------------

        client.train(
            epochs=LOCAL_EPOCHS
        )

        # --------------------------------------------------
        # Get trained parameters
        # --------------------------------------------------

        parameters = (
            client.get_parameters()
        )

        # --------------------------------------------------
        # Calculate client update
        # --------------------------------------------------

        update = client.get_update(
            global_parameters
        )

        # --------------------------------------------------
        # Store client information
        # --------------------------------------------------

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
    # TRUST ANALYSIS
    # ======================================================

    # ------------------------------------------------------
    # PID detector
    # ------------------------------------------------------

    distances, pid_scores = (
        server.detector.calculate_scores(
            client_updates
        )
    )


    # ------------------------------------------------------
    # Relative anomaly
    # ------------------------------------------------------

    relative_anomaly = (
        server.trust_engine.calculate_relative_anomaly(
            pid_scores
        )
    )


    # ------------------------------------------------------
    # Dynamic trust calculation
    #
    # Uses:
    #   Previous trust
    #   Current anomaly
    #   Historical persistence
    # ------------------------------------------------------

    trust_scores = (
        server.trust_engine.calculate_trust(
            pid_scores
        )
    )


    # ------------------------------------------------------
    # Update trust and anomaly history
    # ------------------------------------------------------

    server.trust_engine.update_history(
        trust_scores,
        relative_anomaly
    )


    # ======================================================
    # DISPLAY CLIENT TRUST
    # ======================================================

    print()
    print("Trust Analysis")
    print("------------------------------")

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
            f"Client {client_id}: "
            f"Trust={trust_score:.4f}, "
            f"Zone={zone}"
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
        f"{suspicious_clients}/{NUM_CLIENTS}"
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
        risk_level
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


    print(
        f"Round {round_number} Accuracy: "
        f"{accuracy_percent:.2f}%"
    )


    # ======================================================
    # STORE ROUND RESULTS
    # ======================================================

    row = {

        "round": round_number,

        "accuracy": accuracy,

        "accuracy_percent": accuracy_percent,

        "risk_score": risk_score,

        "risk_level": risk_level,

        "suspicious_clients": suspicious_clients,

        "aggregator": selected_aggregator
    }


    # ------------------------------------------------------
    # Store trust score for every client
    # ------------------------------------------------------

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        row[
            f"trust_client_{client_id}"
        ] = trust_scores[client_id]


    # ------------------------------------------------------
    # Add row to experiment results
    # ------------------------------------------------------

    results.append(row)


# ==========================================================
# SAVE RESULTS TO CSV
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
# EXPERIMENT SUMMARY
# ==========================================================

final_accuracy = (
    results[-1]["accuracy_percent"]
)


best_accuracy = max(
    row["accuracy_percent"]
    for row in results
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print()
print("==============================")
print("Experiment Completed")
print("==============================")


print(
    f"Final Accuracy: "
    f"{final_accuracy:.2f}%"
)


print(
    f"Best Accuracy: "
    f"{best_accuracy:.2f}%"
)


print(
    f"Results saved to: "
    f"{RESULT_FILE}"
)