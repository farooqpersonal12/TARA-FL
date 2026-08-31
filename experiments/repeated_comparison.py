import csv
import os
import random
import statistics

import torch

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist
from attacks.label_flip import LabelFlipDataset


# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

NUM_CLIENTS = 3
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1

MALICIOUS_CLIENTS = [3]
FLIP_RATIO = 0.5

SEEDS = [42, 43, 44, 45, 46]

RESULT_DIR = "experiments"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "repeated_comparison_50pct_label_flip.csv"
)


# ==========================================================
# SEED CONTROL
# ==========================================================

def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)


# ==========================================================
# CREATE CLIENTS
# ==========================================================

def create_experiment_clients(
        train_dataset,
        seed
):

    # ------------------------------------------------------
    # Important:
    # Use a seeded generator so every experiment can be
    # reproduced for the same seed.
    # ------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(seed)

    client_datasets = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) // NUM_CLIENTS] * NUM_CLIENTS,
        generator=generator
    )

    clients = []

    for i in range(NUM_CLIENTS):

        client_id = i + 1

        client_dataset = client_datasets[i]

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

        client = Client(
            client_id=client_id,
            dataset=client_dataset
        )

        clients.append(client)

        print(
            f"Client {client.client_id} created with "
            f"{len(client.dataset)} samples."
        )

    return clients


# ==========================================================
# STANDARD FEDAVG
# ==========================================================

def fedavg(
        client_parameters,
        client_sizes
):

    total_samples = sum(client_sizes)

    new_parameters = {}

    for parameter_name in client_parameters[0]:

        weighted_sum = torch.zeros_like(
            client_parameters[0][parameter_name]
        )

        for parameters, size in zip(
                client_parameters,
                client_sizes
        ):

            weight = size / total_samples

            weighted_sum += (
                    parameters[parameter_name] * weight
            )

        new_parameters[parameter_name] = weighted_sum

    return new_parameters


# ==========================================================
# RUN TARA-FL
# ==========================================================

def run_tara_fl(
        train_dataset,
        test_dataset,
        seed
):

    set_seed(seed)

    print()
    print("########################################")
    print(f"TARA-FL | Seed {seed}")
    print("########################################")

    server = Server()

    clients = create_experiment_clients(
        train_dataset,
        seed
    )

    round_results = []

    for round_number in range(
            1,
            NUM_ROUNDS + 1
    ):

        print()
        print("==============================")
        print(f"Federated Round {round_number}")
        print("==============================")

        global_parameters = (
            server.global_model.state_dict()
        )

        for client in clients:

            client.set_model(
                global_parameters
            )

        client_parameters = []
        client_sizes = []
        client_updates = {}

        # --------------------------------------------------
        # Local training
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Trust analysis
        # --------------------------------------------------

        distances, pid_scores = (
            server.detector.calculate_scores(
                client_updates
            )
        )

        trust_scores = (
            server.trust_engine.calculate_trust(
                pid_scores
            )
        )

        server.trust_engine.update_history(
            trust_scores
        )

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

        # --------------------------------------------------
        # Risk assessment
        # --------------------------------------------------

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
            f"Risk Score: {risk_score:.4f}"
        )

        print(
            f"Risk Level: {risk_level}"
        )

        print(
            f"Suspicious Clients: "
            f"{suspicious_clients}/{NUM_CLIENTS}"
        )

        # --------------------------------------------------
        # Adaptive aggregation
        # --------------------------------------------------

        (
            new_parameters,
            selected_aggregator
        ) = server.aggregate(
            client_parameters,
            client_sizes,
            trust_scores,
            risk_level
        )

        server.global_model.load_state_dict(
            new_parameters
        )

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

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

        round_results.append({
            "round": round_number,
            "accuracy": accuracy_percent,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "suspicious_clients":
                suspicious_clients,
            "aggregator":
                selected_aggregator
        })

    return round_results


# ==========================================================
# RUN STANDARD FEDAVG
# ==========================================================

def run_standard_fedavg(
        train_dataset,
        test_dataset,
        seed
):

    set_seed(seed)

    print()
    print("########################################")
    print(f"STANDARD FEDAVG | Seed {seed}")
    print("########################################")

    server = Server()

    clients = create_experiment_clients(
        train_dataset,
        seed
    )

    round_results = []

    for round_number in range(
            1,
            NUM_ROUNDS + 1
    ):

        print()
        print("==============================")
        print(f"Federated Round {round_number}")
        print("==============================")

        global_parameters = (
            server.global_model.state_dict()
        )

        client_parameters = []
        client_sizes = []

        # --------------------------------------------------
        # Local training
        # --------------------------------------------------

        for client in clients:

            print(
                f"Training Client "
                f"{client.client_id}"
            )

            client.set_model(
                global_parameters
            )

            client.train(
                epochs=LOCAL_EPOCHS
            )

            client_parameters.append(
                client.get_parameters()
            )

            client_sizes.append(
                len(client.dataset)
            )

        # --------------------------------------------------
        # Standard FedAvg
        # --------------------------------------------------

        new_parameters = fedavg(
            client_parameters,
            client_sizes
        )

        server.global_model.load_state_dict(
            new_parameters
        )

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        accuracy = server.evaluate(
            test_dataset
        )

        accuracy_percent = (
                accuracy * 100
        )

        print(
            "Aggregation: STANDARD_FEDAVG"
        )

        print(
            f"Round {round_number} Accuracy: "
            f"{accuracy_percent:.2f}%"
        )

        round_results.append({
            "round": round_number,
            "accuracy": accuracy_percent,
            "aggregator":
                "STANDARD_FEDAVG"
        })

    return round_results


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

train_dataset, test_dataset = load_mnist()

print()
print("========================================")
print("REPEATED COMPARISON")
print("========================================")

all_results = []


for seed in SEEDS:

    # ------------------------------------------------------
    # TARA-FL
    # ------------------------------------------------------

    tara_results = run_tara_fl(
        train_dataset,
        test_dataset,
        seed
    )

    tara_final = (
        tara_results[-1]["accuracy"]
    )

    all_results.append({
        "method": "TARA-FL",
        "seed": seed,
        "final_accuracy": tara_final
    })

    # ------------------------------------------------------
    # Standard FedAvg
    # ------------------------------------------------------

    fedavg_results = run_standard_fedavg(
        train_dataset,
        test_dataset,
        seed
    )

    fedavg_final = (
        fedavg_results[-1]["accuracy"]
    )

    all_results.append({
        "method": "STANDARD_FEDAVG",
        "seed": seed,
        "final_accuracy": fedavg_final
    })


# ==========================================================
# SAVE INDIVIDUAL RESULTS
# ==========================================================

with open(
        RESULT_FILE,
        "w",
        newline=""
) as file:

    fieldnames = [
        "method",
        "seed",
        "final_accuracy"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ==========================================================
# SUMMARY
# ==========================================================

tara_accuracies = [
    row["final_accuracy"]
    for row in all_results
    if row["method"] == "TARA-FL"
]

fedavg_accuracies = [
    row["final_accuracy"]
    for row in all_results
    if row["method"] == "STANDARD_FEDAVG"
]


tara_mean = statistics.mean(
    tara_accuracies
)

tara_std = statistics.stdev(
    tara_accuracies
)

fedavg_mean = statistics.mean(
    fedavg_accuracies
)

fedavg_std = statistics.stdev(
    fedavg_accuracies
)


# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

print()
print("========================================")
print("REPEATED EXPERIMENT SUMMARY")
print("========================================")

print()
print("TARA-FL")
print("----------------------------------------")

print(
    f"Final Accuracies: "
    f"{tara_accuracies}"
)

print(
    f"Mean Accuracy: "
    f"{tara_mean:.4f}%"
)

print(
    f"Standard Deviation: "
    f"{tara_std:.4f}%"
)

print()
print("STANDARD FEDAVG")
print("----------------------------------------")

print(
    f"Final Accuracies: "
    f"{fedavg_accuracies}"
)

print(
    f"Mean Accuracy: "
    f"{fedavg_mean:.4f}%"
)

print(
    f"Standard Deviation: "
    f"{fedavg_std:.4f}%"
)

print()
print(
    f"Results saved to: {RESULT_FILE}"
)