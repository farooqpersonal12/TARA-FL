"""
TARA-FL Unified Experiment Runner.

Single configurable script to run any experiment scenario.

Usage examples:

    # TARA-FL under 20% malicious with label flipping (IID)
    python run_unified.py --method tara --num_clients 20 --malicious_ratio 0.2 --attack label_flip --flip_ratio 0.75 --rounds 30 --seeds 42 123 456 789 1000

    # Standard FedAvg baseline (same attack)
    python run_unified.py --method fedavg --num_clients 20 --malicious_ratio 0.2 --attack label_flip --flip_ratio 0.75 --rounds 30 --seeds 42 123 456 789 1000

    # TARA-FL with gradient poisoning (non-IID)
    python run_unified.py --method tara --num_clients 20 --malicious_ratio 0.2 --attack gradient_scale --scale_factor 10.0 --data_split noniid --alpha 0.5 --rounds 30 --seeds 42 123

    # Clean environment (no attack)
    python run_unified.py --method tara --num_clients 20 --malicious_ratio 0.0 --rounds 30 --seeds 42 123 456 789 1000
"""

import argparse
import csv
import os
import random
import time

import torch

from server.server import Server
from clients.client import Client
from data.dataset import (
    load_mnist,
    create_clients,
    create_clients_noniid
)
from attacks.label_flip import LabelFlipDataset
from attacks.gradient_poison import (
    GradientScaleAttack,
    SignFlipAttack
)


# ==========================================================
# STANDARD FEDAVG (BASELINE)
# ==========================================================

def fedavg(client_parameters, client_sizes):
    """Standard FedAvg — weighted average by dataset size."""

    total_samples = sum(client_sizes)

    new_parameters = {}

    for name in client_parameters[0]:

        weighted_sum = torch.zeros_like(
            client_parameters[0][name]
        )

        for parameters, size in zip(
                client_parameters,
                client_sizes
        ):

            weight = size / total_samples

            weighted_sum += (
                    parameters[name] * weight
            )

        new_parameters[name] = weighted_sum

    return new_parameters


# ==========================================================
# RUN SINGLE EXPERIMENT
# ==========================================================

def run_experiment(config, seed):
    """
    Run one complete federated learning experiment
    with the given configuration and seed.

    Returns a list of per-round result dictionaries.
    """

    # ======================================================
    # REPRODUCIBILITY
    # ======================================================

    random.seed(seed)
    torch.manual_seed(seed)

    # ======================================================
    # SETUP
    # ======================================================

    method = config.method
    num_clients = config.num_clients
    num_rounds = config.rounds
    local_epochs = config.local_epochs

    print()
    print("########################################")
    print(
        f"{method.upper()} | "
        f"Seed {seed} | "
        f"{num_clients} clients"
    )
    print("########################################")

    # ======================================================
    # CREATE SERVER
    # ======================================================

    server = Server(
        detector_type=config.detector_type
    )

    print()
    print("Server created.")
    print(
        f"Method: {method.upper()}"
    )

    if method == "tara":

        print(
            f"Detector: {config.detector_type}"
        )

    # ======================================================
    # LOAD MNIST
    # ======================================================

    train_dataset, test_dataset = load_mnist()

    print("MNIST loaded.")

    # ======================================================
    # CREATE CLIENT DATASETS
    # ======================================================

    if config.data_split == "noniid":

        client_datasets = create_clients_noniid(
            train_dataset,
            num_clients=num_clients,
            alpha=config.alpha,
            seed=seed
        )

        print(
            f"Non-IID split "
            f"(alpha={config.alpha})"
        )

    else:

        client_datasets = create_clients(
            train_dataset,
            num_clients=num_clients
        )

        print("IID split")

    # ======================================================
    # DETERMINE MALICIOUS CLIENTS
    # ======================================================

    num_malicious = int(
        num_clients * config.malicious_ratio
    )

    malicious_ids = list(range(
        num_clients - num_malicious + 1,
        num_clients + 1
    ))

    # ======================================================
    # CREATE CLIENTS
    # ======================================================

    clients = []

    gradient_attacks = {}

    for i in range(num_clients):

        client_id = i + 1

        client_dataset = client_datasets[i]

        # --------------------------------------------------
        # Apply label-flipping attack
        # --------------------------------------------------

        if (
                client_id in malicious_ids
                and config.attack == "label_flip"
        ):

            print(
                f"Applying label-flip attack to "
                f"Client {client_id} "
                f"(flip_ratio={config.flip_ratio})"
            )

            client_dataset = LabelFlipDataset(
                client_dataset,
                flip_ratio=config.flip_ratio,
                seed=seed
            )

        # --------------------------------------------------
        # Prepare gradient attack
        # --------------------------------------------------

        if (
                client_id in malicious_ids
                and config.attack == "gradient_scale"
        ):

            print(
                f"Applying gradient-scale attack to "
                f"Client {client_id} "
                f"(scale={config.scale_factor})"
            )

            gradient_attacks[client_id] = (
                GradientScaleAttack(
                    scale_factor=config.scale_factor
                )
            )

        if (
                client_id in malicious_ids
                and config.attack == "sign_flip"
        ):

            print(
                f"Applying sign-flip attack to "
                f"Client {client_id}"
            )

            gradient_attacks[client_id] = (
                SignFlipAttack()
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
    # FEDERATED TRAINING LOOP
    # ======================================================

    for round_number in range(
            1,
            num_rounds + 1
    ):

        print()
        print("==============================")
        print(
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
        # LOCAL TRAINING
        # ==================================================

        client_parameters = []
        client_sizes = []
        client_updates = {}

        for client in clients:

            print(
                f"Training Client "
                f"{client.client_id}"
            )

            client.set_model(
                global_parameters
            )

            client.train(
                epochs=local_epochs
            )

            parameters = (
                client.get_parameters()
            )

            update = client.get_update(
                global_parameters
            )

            # ------------------------------------------
            # Apply gradient attack if configured
            # ------------------------------------------

            if client.client_id in gradient_attacks:

                update = gradient_attacks[
                    client.client_id
                ].apply(update)

                # Reconstruct poisoned parameters
                # from global + poisoned update.
                poisoned_parameters = {}

                for name in global_parameters:

                    poisoned_parameters[name] = (
                            global_parameters[name]
                            + update[name]
                    )

                parameters = poisoned_parameters

            client_parameters.append(
                parameters
            )

            client_sizes.append(
                len(client.dataset)
            )

            client_updates[
                client.client_id
            ] = update

        # ==================================================
        # METHOD-SPECIFIC AGGREGATION
        # ==================================================

        if method == "fedavg":

            # ----------------------------------------------
            # Standard FedAvg — no defense
            # ----------------------------------------------

            new_parameters = fedavg(
                client_parameters,
                client_sizes
            )

            selected_aggregator = "STANDARD_FEDAVG"

            risk_score = 0.0
            risk_level = "N/A"
            suspicious_clients = 0
            trust_scores = {}
            pid_scores = {}
            distances = {}

        else:

            # ----------------------------------------------
            # TARA-FL — full pipeline
            # ----------------------------------------------

            # PID anomaly detection
            distances, pid_scores = (
                server.detector.calculate_scores(
                    client_updates
                )
            )

            # Adaptive trust analysis
            all_pid_scores = list(
                pid_scores.values()
            )

            trust_scores = {}
            trust_details = {}

            for client_id, pid_score in (
                    pid_scores.items()
            ):

                result = (
                    server.trust_engine.calculate_trust(
                        client_id,
                        pid_score,
                        all_pid_scores
                    )
                )

                trust_scores[client_id] = (
                    result["trust"]
                )

                trust_details[client_id] = result

            # Display trust analysis
            print()
            print("Trust Analysis")
            print("------------------------------")

            for client in clients:

                cid = client.client_id
                details = trust_details[cid]

                print(
                    f"Client {cid}: "
                    f"PID={details['pid_score']:.4f} "
                    f"Trust={details['trust']:.4f}"
                )

            # Round risk
            (
                risk_score,
                risk_level,
                suspicious_clients
            ) = server.round_risk.calculate_risk(
                distances,
                trust_scores
            )

            print()
            print(
                f"Round Risk: "
                f"{risk_score:.4f} "
                f"({risk_level})"
            )

            print(
                f"Suspicious Clients: "
                f"{suspicious_clients}/"
                f"{num_clients}"
            )

            # Adaptive aggregation
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

        accuracy_percent = accuracy * 100

        print()
        print(
            f"Round {round_number} Accuracy: "
            f"{accuracy_percent:.2f}%"
        )

        print(
            f"Aggregator: "
            f"{selected_aggregator}"
        )

        # ==================================================
        # STORE ROUND RESULTS
        # ==================================================

        row = {
            "method": method,
            "seed": seed,
            "round": round_number,
            "accuracy": accuracy,
            "accuracy_percent": accuracy_percent,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "suspicious_clients":
                suspicious_clients,
            "aggregator":
                selected_aggregator
        }

        # Store per-client trust scores.
        for cid in range(1, num_clients + 1):

            row[f"trust_client_{cid}"] = (
                trust_scores.get(cid, "N/A")
            )

        results.append(row)

    return results


# ==========================================================
# ARGUMENT PARSER
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="TARA-FL Unified Experiment Runner"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="tara",
        choices=["tara", "fedavg"],
        help="Aggregation method"
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="Number of FL clients"
    )

    parser.add_argument(
        "--malicious_ratio",
        type=float,
        default=0.2,
        help="Fraction of malicious clients"
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="label_flip",
        choices=[
            "none",
            "label_flip",
            "gradient_scale",
            "sign_flip"
        ],
        help="Attack type"
    )

    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=0.75,
        help="Label flip ratio (for label_flip attack)"
    )

    parser.add_argument(
        "--scale_factor",
        type=float,
        default=10.0,
        help="Scale factor (for gradient_scale attack)"
    )

    parser.add_argument(
        "--data_split",
        type=str,
        default="iid",
        choices=["iid", "noniid"],
        help="Data partitioning strategy"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for non-IID split"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated rounds"
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Local training epochs per round"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1000],
        help="Random seeds for repeated runs"
    )

    parser.add_argument(
        "--detector_type",
        type=str,
        default="standard",
        choices=["standard", "robust"],
        help="PID detector type"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Output directory for results"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name for output file"
    )

    return parser.parse_args()


# ==========================================================
# MAIN
# ==========================================================

def main():

    config = parse_args()

    # ======================================================
    # CREATE OUTPUT DIRECTORY
    # ======================================================

    os.makedirs(
        config.output_dir,
        exist_ok=True
    )

    # ======================================================
    # GENERATE EXPERIMENT NAME
    # ======================================================

    if config.experiment_name:

        experiment_name = config.experiment_name

    else:

        num_malicious = int(
            config.num_clients
            * config.malicious_ratio
        )

        experiment_name = (
            f"{config.method}_"
            f"{config.num_clients}clients_"
            f"{num_malicious}mal_"
            f"{config.attack}_"
            f"{config.data_split}"
        )

    result_file = os.path.join(
        config.output_dir,
        f"results_{experiment_name}.csv"
    )

    summary_file = os.path.join(
        config.output_dir,
        f"summary_{experiment_name}.csv"
    )

    # ======================================================
    # DISPLAY CONFIGURATION
    # ======================================================

    print()
    print("==========================================")
    print("TARA-FL UNIFIED EXPERIMENT RUNNER")
    print("==========================================")

    print()
    print("Configuration")
    print("------------------------------------------")

    print(f"Method:           {config.method}")
    print(f"Clients:          {config.num_clients}")
    print(f"Malicious Ratio:  {config.malicious_ratio}")
    print(f"Attack:           {config.attack}")
    print(f"Data Split:       {config.data_split}")
    print(f"Rounds:           {config.rounds}")
    print(f"Local Epochs:     {config.local_epochs}")
    print(f"Seeds:            {config.seeds}")
    print(f"Detector:         {config.detector_type}")

    if config.attack == "label_flip":

        print(
            f"Flip Ratio:       "
            f"{config.flip_ratio}"
        )

    if config.attack == "gradient_scale":

        print(
            f"Scale Factor:     "
            f"{config.scale_factor}"
        )

    if config.data_split == "noniid":

        print(
            f"Alpha:            "
            f"{config.alpha}"
        )

    # ======================================================
    # RUN ALL SEEDS
    # ======================================================

    all_results = []
    seed_summaries = []

    start_time = time.time()

    for seed in config.seeds:

        seed_results = run_experiment(
            config,
            seed
        )

        all_results.extend(seed_results)

        final_accuracy = (
            seed_results[-1]["accuracy_percent"]
        )

        best_accuracy = max(
            row["accuracy_percent"]
            for row in seed_results
        )

        seed_summaries.append({
            "seed": seed,
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy
        })

    total_time = time.time() - start_time

    # ======================================================
    # SAVE ROUND RESULTS
    # ======================================================

    if all_results:

        fieldnames = list(
            all_results[0].keys()
        )

        with open(
                result_file,
                "w",
                newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(all_results)

    # ======================================================
    # COMPUTE STATISTICS
    # ======================================================

    final_accuracies = [
        row["final_accuracy"]
        for row in seed_summaries
    ]

    best_accuracies = [
        row["best_accuracy"]
        for row in seed_summaries
    ]

    mean_final = (
            sum(final_accuracies)
            / len(final_accuracies)
    )

    mean_best = (
            sum(best_accuracies)
            / len(best_accuracies)
    )

    if len(final_accuracies) > 1:

        final_variance = sum(
            (acc - mean_final) ** 2
            for acc in final_accuracies
        ) / (len(final_accuracies) - 1)

        final_std = final_variance ** 0.5

        best_variance = sum(
            (acc - mean_best) ** 2
            for acc in best_accuracies
        ) / (len(best_accuracies) - 1)

        best_std = best_variance ** 0.5

    else:

        final_std = 0.0
        best_std = 0.0

    # ======================================================
    # SAVE SUMMARY
    # ======================================================

    with open(
            summary_file,
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
        writer.writerows(seed_summaries)

        writer.writerow({
            "seed": "MEAN",
            "final_accuracy": mean_final,
            "best_accuracy": mean_best
        })

        writer.writerow({
            "seed": "STD",
            "final_accuracy": final_std,
            "best_accuracy": best_std
        })

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print()
    print("==========================================")
    print("EXPERIMENT COMPLETED")
    print("==========================================")

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
        f"{mean_final:.2f}%"
    )

    print(
        f"Std Final Accuracy:  "
        f"{final_std:.2f}%"
    )

    print(
        f"Mean Best Accuracy:  "
        f"{mean_best:.2f}%"
    )

    print(
        f"Std Best Accuracy:   "
        f"{best_std:.2f}%"
    )

    print(
        f"Total Time:          "
        f"{total_time:.1f}s"
    )

    print()
    print(f"Results saved to: {result_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":

    main()
