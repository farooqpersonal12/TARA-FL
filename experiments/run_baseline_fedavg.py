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

# ----------------------------------------------------------
# EXP-17
#
# 10 clients
# 2 malicious clients
# 75% label flipping
# Standard FedAvg baseline
#
# This experiment is used as the baseline comparison
# against the TARA-FL adaptive aggregation approach.
# ----------------------------------------------------------

NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
SEED = 42

MALICIOUS_CLIENTS = [9, 10]
FLIP_RATIO = 0.75

RESULT_DIR = "experiments"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "results_exp17_standard_fedavg_10clients_2malicious_75pct.csv"
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
print("Using Standard FedAvg.")


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
    # Apply label-flipping attack to malicious clients
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
# STANDARD FEDAVG
# ==========================================================

def fedavg(
        client_parameters,
        client_sizes
):

    total_samples = sum(
        client_sizes
    )

    new_parameters = {}

    for parameter_name in client_parameters[0]:

        weighted_sum = torch.zeros_like(
            client_parameters[0][
                parameter_name
            ]
        )

        for parameters, size in zip(
                client_parameters,
                client_sizes
        ):

            weight = (
                    size / total_samples
            )

            weighted_sum += (
                    parameters[
                        parameter_name
                    ] * weight
            )

        new_parameters[
            parameter_name
        ] = weighted_sum

    return new_parameters


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
        f"Federated Round "
        f"{round_number}"
    )
    print("==============================")


    # ======================================================
    # GET GLOBAL MODEL
    # ======================================================

    global_parameters = (
        server.global_model.state_dict()
    )


    # ======================================================
    # CLIENT CONTAINERS
    # ======================================================

    client_parameters = []
    client_sizes = []


    # ======================================================
    # LOCAL TRAINING
    # ======================================================

    for client in clients:

        print(
            f"Training Client "
            f"{client.client_id}"
        )

        # --------------------------------------------------
        # Send global model to client
        # --------------------------------------------------

        client.set_model(
            global_parameters
        )

        # --------------------------------------------------
        # Local training
        # --------------------------------------------------

        client.train(
            epochs=LOCAL_EPOCHS
        )

        # --------------------------------------------------
        # Get trained client parameters
        # --------------------------------------------------

        client_parameters.append(
            client.get_parameters()
        )

        # --------------------------------------------------
        # Get client dataset size
        # --------------------------------------------------

        client_sizes.append(
            len(client.dataset)
        )


    # ======================================================
    # STANDARD FEDAVG AGGREGATION
    # ======================================================

    new_parameters = fedavg(
        client_parameters,
        client_sizes
    )

    server.global_model.load_state_dict(
        new_parameters
    )


    print()
    print(
        "Aggregation: "
        "STANDARD_FEDAVG"
    )


    # ======================================================
    # GLOBAL EVALUATION
    # ======================================================

    accuracy = server.evaluate(
        test_dataset
    )

    accuracy_percent = (
            accuracy * 100
    )


    print(
        f"Round {round_number} "
        f"Accuracy: "
        f"{accuracy_percent:.2f}%"
    )


    # ======================================================
    # STORE ROUND RESULTS
    # ======================================================

    results.append({

        "round": round_number,

        "accuracy": accuracy,

        "accuracy_percent":
            accuracy_percent,

        "aggregator":
            "STANDARD_FEDAVG"
    })


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
# EXPERIMENT SUMMARY
# ==========================================================

final_accuracy = (
    results[-1]["accuracy_percent"]
)

best_accuracy = max(
    row["accuracy_percent"]
    for row in results
)


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