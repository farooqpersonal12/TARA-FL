import csv
import os
import random
import torch

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients
from attacks.label_flip import LabelFlipDataset


# ==========================================================
# EXPERIMENT 19
# ==========================================================
#
# Multi-seed Standard FedAvg Baseline
#
# Configuration:
#   10 clients
#   2 malicious clients
#   75% label flipping
#   10 federated rounds
#   5 independent seeds
#
# This experiment is the baseline comparison for
# Experiment 18 (TARA-FL multi-seed).
#
# IMPORTANT:
# Standard FedAvg is used WITHOUT:
#   - PID detection
#   - Trust scoring
#   - Risk assessment
#   - Adaptive aggregation
#
# ==========================================================


# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1

SEEDS = [
    42,
    123,
    456,
    789,
    1000
]

MALICIOUS_CLIENTS = [
    9,
    10
]

FLIP_RATIO = 0.75


# ==========================================================
# RESULT DIRECTORY
# ==========================================================

RESULT_DIR = "experiments"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ==========================================================
# RESULT FILES
# ==========================================================

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "results_exp19_fedavg_multi_seed_10clients_2malicious_75pct.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "summary_exp19_fedavg_multi_seed_10clients_2malicious_75pct.csv"
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
# RUN ONE SEED
# ==========================================================

def run_experiment(seed):

    print()
    print("########################################")
    print(
        f"STARTING SEED {seed}"
    )
    print("########################################")


    # ======================================================
    # REPRODUCIBILITY
    # ======================================================

    random.seed(seed)
    torch.manual_seed(seed)


    # ======================================================
    # CREATE SERVER
    # ======================================================

    server = Server()

    print()
    print("Server created.")
    print("Using Standard FedAvg.")


    # ======================================================
    # LOAD MNIST
    # ======================================================

    train_dataset, test_dataset = (
        load_mnist()
    )

    print("MNIST loaded.")


    # ======================================================
    # CREATE CLIENT DATASETS
    # ======================================================

    client_datasets = create_clients(
        train_dataset,
        num_clients=NUM_CLIENTS
    )


    # ======================================================
    # CREATE CLIENTS
    # ======================================================

    clients = []

    for i in range(NUM_CLIENTS):

        client_id = i + 1

        client_dataset = (
            client_datasets[i]
        )


        # --------------------------------------------------
        # Apply label-flipping attack
        # --------------------------------------------------

        if client_id in MALICIOUS_CLIENTS:

            print(
                f"Applying label-flip attack "
                f"to Client {client_id}"
            )

            client_dataset = (
                LabelFlipDataset(
                    client_dataset,
                    flip_ratio=FLIP_RATIO,
                    seed=seed
                )
            )


        # --------------------------------------------------
        # Create client
        # --------------------------------------------------

        client = Client(
            client_id=client_id,
            dataset=client_dataset
        )

        clients.append(
            client
        )


        print(
            f"Client {client.client_id} "
            f"created with "
            f"{len(client.dataset)} samples."
        )


    # ======================================================
    # RESULTS STORAGE
    # ======================================================

    results = []


    # ======================================================
    # FEDERATED TRAINING
    # ======================================================

    for round_number in range(
            1,
            NUM_ROUNDS + 1
    ):

        print()
        print("==============================")
        print(
            f"Seed {seed} | "
            f"Federated Round "
            f"{round_number}"
        )
        print("==============================")


        # ==================================================
        # GET GLOBAL PARAMETERS
        # ==================================================

        global_parameters = (
            server.global_model.state_dict()
        )


        # ==================================================
        # CLIENT CONTAINERS
        # ==================================================

        client_parameters = []
        client_sizes = []


        # ==================================================
        # LOCAL TRAINING
        # ==================================================

        for client in clients:

            print(
                f"Training Client "
                f"{client.client_id}"
            )


            # ------------------------------------------------
            # Send global model
            # ------------------------------------------------

            client.set_model(
                global_parameters
            )


            # ------------------------------------------------
            # Local training
            # ------------------------------------------------

            client.train(
                epochs=LOCAL_EPOCHS
            )


            # ------------------------------------------------
            # Collect parameters
            # ------------------------------------------------

            client_parameters.append(
                client.get_parameters()
            )


            # ------------------------------------------------
            # Collect client size
            # ------------------------------------------------

            client_sizes.append(
                len(client.dataset)
            )


        # ==================================================
        # STANDARD FEDAVG
        # ==================================================

        new_parameters = fedavg(
            client_parameters,
            client_sizes
        )


        # ==================================================
        # UPDATE GLOBAL MODEL
        # ==================================================

        server.global_model.load_state_dict(
            new_parameters
        )


        # ==================================================
        # GLOBAL EVALUATION
        # ==================================================

        accuracy = server.evaluate(
            test_dataset
        )

        accuracy_percent = (
                accuracy * 100
        )


        # ==================================================
        # DISPLAY RESULT
        # ==================================================

        print()
        print(
            "Aggregation: "
            "STANDARD_FEDAVG"
        )

        print(
            f"Round {round_number} "
            f"Accuracy: "
            f"{accuracy_percent:.2f}%"
        )


        # ==================================================
        # STORE ROUND RESULT
        # ==================================================

        results.append({

            "seed":
                seed,

            "round":
                round_number,

            "accuracy":
                accuracy,

            "accuracy_percent":
                accuracy_percent,

            "aggregator":
                "STANDARD_FEDAVG"
        })


    # ======================================================
    # SEED SUMMARY
    # ======================================================

    final_accuracy = (
        results[-1]["accuracy_percent"]
    )

    best_accuracy = max(
        row["accuracy_percent"]
        for row in results
    )


    print()
    print("----------------------------------------")
    print(
        f"Seed {seed} Completed"
    )
    print("----------------------------------------")

    print(
        f"Final Accuracy: "
        f"{final_accuracy:.2f}%"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}%"
    )


    return (
        results,
        final_accuracy,
        best_accuracy
    )


# ==========================================================
# RUN ALL SEEDS
# ==========================================================

all_results = []
seed_summaries = []


for seed in SEEDS:

    (
        seed_results,
        final_accuracy,
        best_accuracy
    ) = run_experiment(seed)


    # ------------------------------------------------------
    # Add round results
    # ------------------------------------------------------

    all_results.extend(
        seed_results
    )


    # ------------------------------------------------------
    # Add seed summary
    # ------------------------------------------------------

    seed_summaries.append({

        "seed":
            seed,

        "final_accuracy":
            final_accuracy,

        "best_accuracy":
            best_accuracy
    })


# ==========================================================
# SAVE ROUND RESULTS
# ==========================================================

if all_results:

    fieldnames = all_results[0].keys()

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
            all_results
        )


# ==========================================================
# MULTI-SEED STATISTICS
# ==========================================================

final_accuracies = [
    row["final_accuracy"]
    for row in seed_summaries
]

best_accuracies = [
    row["best_accuracy"]
    for row in seed_summaries
]


# ----------------------------------------------------------
# Mean
# ----------------------------------------------------------

mean_final_accuracy = (
        sum(final_accuracies)
        / len(final_accuracies)
)

mean_best_accuracy = (
        sum(best_accuracies)
        / len(best_accuracies)
)


# ----------------------------------------------------------
# Standard deviation
#
# Use population standard deviation because these
# configured seeds represent the complete set of runs
# for this experiment.
# ----------------------------------------------------------

final_variance = sum(
    (
            accuracy - mean_final_accuracy
    ) ** 2
    for accuracy in final_accuracies
) / len(final_accuracies)

best_variance = sum(
    (
            accuracy - mean_best_accuracy
    ) ** 2
    for accuracy in best_accuracies
) / len(best_accuracies)


final_std = (
        final_variance ** 0.5
)

best_std = (
        best_variance ** 0.5
)


# ==========================================================
# SAVE SEED SUMMARY
# ==========================================================

summary_fieldnames = [
    "seed",
    "final_accuracy",
    "best_accuracy"
]


with open(
        SUMMARY_FILE,
        "w",
        newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary_fieldnames
    )

    writer.writeheader()

    writer.writerows(
        seed_summaries
    )


# ==========================================================
# SAVE MULTI-SEED STATISTICS
# ==========================================================

statistics_file = os.path.join(
    RESULT_DIR,
    "statistics_exp19_fedavg_multi_seed_10clients_2malicious_75pct.csv"
)


with open(
        statistics_file,
        "w",
        newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "metric",
            "mean",
            "std"
        ]
    )

    writer.writeheader()

    writer.writerow({
        "metric":
            "final_accuracy",

        "mean":
            mean_final_accuracy,

        "std":
            final_std
    })

    writer.writerow({
        "metric":
            "best_accuracy",

        "mean":
            mean_best_accuracy,

        "std":
            best_std
    })


# ==========================================================
# FINAL EXPERIMENT SUMMARY
# ==========================================================

print()
print("==========================================")
print("EXPERIMENT 19 COMPLETED")
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
    f"Local Epochs: "
    f"{LOCAL_EPOCHS}"
)

print(
    f"Seeds: "
    f"{SEEDS}"
)

print()
print("Per-Seed Results")
print("------------------------------------------")


for row in seed_summaries:

    print(
        f"Seed {row['seed']}: "
        f"Final={row['final_accuracy']:.2f}%, "
        f"Best={row['best_accuracy']:.2f}%"
    )


print()
print("Multi-Seed Statistics")
print("------------------------------------------")

print(
    f"Mean Final Accuracy: "
    f"{mean_final_accuracy:.2f}%"
)

print(
    f"Std Final Accuracy: "
    f"{final_std:.2f}%"
)

print(
    f"Mean Best Accuracy: "
    f"{mean_best_accuracy:.2f}%"
)

print(
    f"Std Best Accuracy: "
    f"{best_std:.2f}%"
)


print()
print(
    f"Round results saved to: "
    f"{RESULT_FILE}"
)

print(
    f"Seed summary saved to: "
    f"{SUMMARY_FILE}"
)

print(
    f"Statistics saved to: "
    f"{statistics_file}"
)