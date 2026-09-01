import csv
import os
import random
import statistics
import torch

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients
from attacks.label_flip import LabelFlipDataset


# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

# ----------------------------------------------------------
# EXP-18
#
# Multi-seed TARA-FL validation
#
# 10 clients
# 2 malicious clients
# 75% label flipping
# Original PID Detector
# Improved Trust Engine
# Improved Round Risk
# Adaptive Aggregation
#
# Only the random seed changes between runs.
# ----------------------------------------------------------

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
    "results_exp18_tara_multi_seed_10clients_2malicious_75pct.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "summary_exp18_tara_multi_seed_10clients_2malicious_75pct.csv"
)


# ==========================================================
# RUN ONE EXPERIMENT
# ==========================================================

def run_experiment(seed):

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
    print("Using Original PID Detector.")
    print("Using Improved Trust Engine.")
    print("Using Improved Round Risk.")
    print("Using Adaptive Aggregation.")


    # ======================================================
    # LOAD MNIST
    # ======================================================

    train_dataset, test_dataset = load_mnist()

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

        client_dataset = client_datasets[i]


        # --------------------------------------------------
        # Apply label-flipping attack
        # --------------------------------------------------

        if client_id in MALICIOUS_CLIENTS:

            print(
                f"Applying label-flip attack to "
                f"Client {client_id}"
            )

            client_dataset = LabelFlipDataset(
                client_dataset,
                flip_ratio=FLIP_RATIO,
                seed=seed
            )


        # --------------------------------------------------
        # Create client
        # --------------------------------------------------

        client = Client(
            client_id=client_id,
            dataset=client_dataset
        )

        clients.append(client)

        print(
            f"Client {client.client_id} created with "
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
            f"Federated Round {round_number}"
        )
        print("==============================")


        # ==================================================
        # GET GLOBAL MODEL PARAMETERS
        # ==================================================

        global_parameters = (
            server.global_model.state_dict()
        )


        # ==================================================
        # SEND GLOBAL MODEL TO CLIENTS
        # ==================================================

        for client in clients:

            client.set_model(
                global_parameters
            )


        # ==================================================
        # CONTAINERS
        # ==================================================

        client_parameters = []

        client_sizes = []

        client_updates = {}


        # ==================================================
        # LOCAL TRAINING
        # ==================================================

        for client in clients:

            print(
                f"Training Client "
                f"{client.client_id}"
            )


            # ------------------------------------------------
            # Local training
            # ------------------------------------------------

            client.train(
                epochs=LOCAL_EPOCHS
            )


            # ------------------------------------------------
            # Get trained parameters
            # ------------------------------------------------

            parameters = (
                client.get_parameters()
            )


            # ------------------------------------------------
            # Calculate update
            # ------------------------------------------------

            update = client.get_update(
                global_parameters
            )


            # ------------------------------------------------
            # Store parameters
            # ------------------------------------------------

            client_parameters.append(
                parameters
            )


            # ------------------------------------------------
            # Store client size
            # ------------------------------------------------

            client_sizes.append(
                len(client.dataset)
            )


            # ------------------------------------------------
            # Store client update
            # ------------------------------------------------

            client_updates[
                client.client_id
            ] = update


        # ==================================================
        # TRUST ANALYSIS
        # ==================================================

        distances, pid_scores = (
            server.detector.calculate_scores(
                client_updates
            )
        )


        # --------------------------------------------------
        # Calculate relative anomaly
        # --------------------------------------------------

        relative_anomaly = (
            server.trust_engine.calculate_relative_anomaly(
                pid_scores
            )
        )


        # --------------------------------------------------
        # Calculate dynamic trust
        # --------------------------------------------------

        trust_scores = (
            server.trust_engine.calculate_trust(
                pid_scores
            )
        )


        # --------------------------------------------------
        # Update trust history
        # --------------------------------------------------

        server.trust_engine.update_history(
            trust_scores,
            relative_anomaly
        )


        # ==================================================
        # DISPLAY TRUST ANALYSIS
        # ==================================================

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


        # ==================================================
        # ROUND RISK
        # ==================================================

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


        # ==================================================
        # ADAPTIVE AGGREGATION
        # ==================================================

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


        # ==================================================
        # UPDATE GLOBAL MODEL
        # ==================================================

        server.global_model.load_state_dict(
            new_parameters
        )


        # ==================================================
        # GLOBAL MODEL EVALUATION
        # ==================================================

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


        # ==================================================
        # STORE ROUND RESULTS
        # ==================================================

        row = {

            "seed": seed,

            "round": round_number,

            "accuracy": accuracy,

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


        # --------------------------------------------------
        # Store trust scores
        # --------------------------------------------------

        for client_id in range(
                1,
                NUM_CLIENTS + 1
        ):

            row[
                f"trust_client_{client_id}"
            ] = trust_scores[client_id]


        results.append(row)


    # ======================================================
    # RETURN RESULTS FOR THIS SEED
    # ======================================================

    return results


# ==========================================================
# MULTI-SEED EXPERIMENT
# ==========================================================

all_results = []

seed_summary = []


for seed in SEEDS:

    print()
    print("########################################")
    print(
        f"STARTING SEED {seed}"
    )
    print("########################################")


    seed_results = run_experiment(
        seed
    )


    # ------------------------------------------------------
    # Add results to combined results
    # ------------------------------------------------------

    all_results.extend(
        seed_results
    )


    # ------------------------------------------------------
    # Calculate seed statistics
    # ------------------------------------------------------

    final_accuracy = (
        seed_results[-1]["accuracy_percent"]
    )

    best_accuracy = max(
        row["accuracy_percent"]
        for row in seed_results
    )


    seed_summary.append({

        "seed":
            seed,

        "final_accuracy":
            final_accuracy,

        "best_accuracy":
            best_accuracy
    })


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


# ==========================================================
# SAVE ALL ROUND RESULTS
# ==========================================================

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
    for row in seed_summary
]

best_accuracies = [
    row["best_accuracy"]
    for row in seed_summary
]


mean_final_accuracy = (
    statistics.mean(
        final_accuracies
    )
)

std_final_accuracy = (
    statistics.stdev(
        final_accuracies
    )
    if len(final_accuracies) > 1
    else 0.0
)


mean_best_accuracy = (
    statistics.mean(
        best_accuracies
    )
)

std_best_accuracy = (
    statistics.stdev(
        best_accuracies
    )
    if len(best_accuracies) > 1
    else 0.0
)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary_rows = []


for row in seed_summary:

    summary_rows.append({

        "seed":
            row["seed"],

        "final_accuracy":
            row["final_accuracy"],

        "best_accuracy":
            row["best_accuracy"]
    })


# ----------------------------------------------------------
# Add aggregate statistics
# ----------------------------------------------------------

summary_rows.append({

    "seed":
        "MEAN",

    "final_accuracy":
        mean_final_accuracy,

    "best_accuracy":
        mean_best_accuracy
})


summary_rows.append({

    "seed":
        "STD",

    "final_accuracy":
        std_final_accuracy,

    "best_accuracy":
        std_best_accuracy
})


with open(
        SUMMARY_FILE,
        "w",
        newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "seed",
            "final_accuracy",
            "best_accuracy"
        ]
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ==========================================================
# FINAL EXPERIMENT SUMMARY
# ==========================================================

print()
print("==========================================")
print("EXPERIMENT 18 COMPLETED")
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
    f"Seeds: "
    f"{SEEDS}"
)


print()
print("Per-Seed Results")
print("------------------------------------------")


for row in seed_summary:

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
    f"{std_final_accuracy:.2f}%"
)

print(
    f"Mean Best Accuracy: "
    f"{mean_best_accuracy:.2f}%"
)

print(
    f"Std Best Accuracy: "
    f"{std_best_accuracy:.2f}%"
)


print()
print(
    f"Round results saved to: "
    f"{RESULT_FILE}"
)

print(
    f"Summary saved to: "
    f"{SUMMARY_FILE}"
)