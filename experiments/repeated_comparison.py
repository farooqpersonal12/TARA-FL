"""
TARA-FL Repeated Multi-Seed Controlled Comparison.

Runs a multi-seed controlled comparison across all 4 baseline and TARA-FL variants:
1. STANDARD_FEDAVG: Base FedAvg (no trust, no defense)
2. TRUST_FEDAVG: Trust Analysis -> Trust-Aware FedAvg
3. TRUST_ROBUST: Trust Analysis -> Fixed Robust Aggregation (Trimming)
4. TARA-FL: Trust Analysis -> Trust-Aware Quarantine -> Round Risk -> Adaptive Aggregation
"""

import csv
import json
import os
import sys
import random
import statistics
import time
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist
from attacks.label_flip import LabelFlipDataset


# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

NUM_CLIENTS = 10
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1

MALICIOUS_CLIENTS = [9, 10]
FLIP_RATIO = 0.75

SEEDS = [42, 43, 44]

RESULT_DIR = "experiments"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "repeated_comparison_4methods.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "repeated_comparison_summary.json"
)


# ==========================================================
# SEED CONTROL
# ==========================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


# ==========================================================
# CREATE CLIENTS
# ==========================================================

def create_experiment_clients(train_dataset, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    split_sizes = [len(train_dataset) // NUM_CLIENTS] * NUM_CLIENTS
    remainder = len(train_dataset) - sum(split_sizes)
    split_sizes[0] += remainder

    client_datasets = torch.utils.data.random_split(
        train_dataset,
        split_sizes,
        generator=generator
    )

    clients = []
    for i in range(NUM_CLIENTS):
        client_id = i + 1
        client_dataset = client_datasets[i]

        if client_id in MALICIOUS_CLIENTS:
            client_dataset = LabelFlipDataset(
                client_dataset,
                flip_ratio=FLIP_RATIO,
                seed=seed
            )

        client = Client(client_id=client_id, dataset=client_dataset)
        clients.append(client)

    return clients


# ==========================================================
# STANDARD FEDAVG HELPER
# ==========================================================

def fedavg(client_parameters, client_sizes):
    total_samples = sum(client_sizes)
    new_parameters = {}
    for name in client_parameters[0]:
        weighted_sum = torch.zeros_like(client_parameters[0][name])
        for parameters, size in zip(client_parameters, client_sizes):
            weight = size / max(1, total_samples)
            weighted_sum += parameters[name] * weight
        new_parameters[name] = weighted_sum
    return new_parameters


# ==========================================================
# METHOD RUNNERS
# ==========================================================

def run_method(method_name: str, train_dataset, test_dataset, seed: int) -> List[Dict[str, Any]]:
    set_seed(seed)

    print()
    print("########################################")
    print(f"{method_name} | Seed {seed}")
    print("########################################")

    enable_quarantine = (method_name == "TARA-FL")
    server = Server(enable_quarantine=enable_quarantine)
    clients = create_experiment_clients(train_dataset, seed)

    round_results = []
    last_accuracy = None

    for round_number in range(1, NUM_ROUNDS + 1):
        round_start_time = time.time()
        global_parameters = server.global_model.state_dict()

        client_parameters = []
        client_sizes = []
        client_updates = {}

        for client in clients:
            client.set_model(global_parameters)
            client.train(epochs=LOCAL_EPOCHS)
            params = client.get_parameters()
            update = client.get_update(global_parameters)

            client_parameters.append(params)
            client_sizes.append(len(client.dataset))
            client_updates[client.client_id] = update

        if method_name == "STANDARD_FEDAVG":
            new_parameters = fedavg(client_parameters, client_sizes)
            selected_aggregator = "STANDARD_FEDAVG"
            risk_score = 0.0
            risk_level = "N/A"
            threat_type = "N/A"
            suspicious_clients = 0
            total_samples = sum(client_sizes)
            malicious_weight = sum(client_sizes[cid - 1] for cid in MALICIOUS_CLIENTS) / total_samples

        elif method_name == "TRUST_FEDAVG":
            distances, pid_scores = server.detector.calculate_scores(client_updates)
            all_pid = list(pid_scores.values())
            trust_scores = {cid: server.trust_engine.calculate_trust(cid, s, all_pid)["trust"] for cid, s in pid_scores.items()}
            new_parameters = server.adaptive_aggregator.strategy_fedavg.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_ids=list(range(1, NUM_CLIENTS + 1))
            )
            selected_aggregator = "TRUST_AWARE_FEDAVG"
            risk_score = 0.0
            risk_level = "N/A"
            threat_type = "N/A"
            suspicious_clients = sum(1 for t in trust_scores.values() if t < 0.75)
            weights = server.adaptive_aggregator.strategy_fedavg.calculate_trust_weights(client_sizes, trust_scores)
            malicious_weight = sum(weights[cid - 1] for cid in MALICIOUS_CLIENTS)

        elif method_name == "TRUST_ROBUST":
            distances, pid_scores = server.detector.calculate_scores(client_updates)
            all_pid = list(pid_scores.values())
            trust_scores = {cid: server.trust_engine.calculate_trust(cid, s, all_pid)["trust"] for cid, s in pid_scores.items()}
            new_parameters = server.adaptive_aggregator.strategy_robust.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_updates=client_updates,
                risk_level="MEDIUM",
                client_ids=list(range(1, NUM_CLIENTS + 1))
            )
            selected_aggregator = "TRUST_WEIGHTED_ROBUST"
            risk_score = 0.0
            risk_level = "N/A"
            threat_type = "N/A"
            suspicious_clients = sum(1 for t in trust_scores.values() if t < 0.75)
            distances_map = server.adaptive_aggregator.calculate_update_distances(client_updates)
            surviving = server.adaptive_aggregator.select_robust_clients(list(range(1, NUM_CLIENTS + 1)), distances_map, "MEDIUM")
            surv_sizes = [client_sizes[cid - 1] for cid in surviving]
            surv_trust = {cid: trust_scores.get(cid, 0.0) for cid in surviving}
            surv_weights = server.adaptive_aggregator.calculate_trust_weights(surv_sizes, surv_trust)
            malicious_weight = sum(w for cid, w in zip(surviving, surv_weights) if cid in MALICIOUS_CLIENTS)

        elif method_name == "TARA-FL":
            distances, pid_scores = server.detector.calculate_scores(client_updates)
            all_pid = list(pid_scores.values())
            trust_scores = {cid: server.trust_engine.calculate_trust(cid, s, all_pid)["trust"] for cid, s in pid_scores.items()}
            risk_score, risk_level, suspicious_clients = server.round_risk.calculate_risk(
                distances=distances,
                trust_scores=trust_scores,
                current_accuracy=last_accuracy,
                client_updates=client_updates
            )
            threat_type = server.round_risk.classify_threat(
                distances=distances,
                trust_scores=trust_scores,
                client_updates=client_updates
            ).value
            new_parameters, selected_aggregator = server.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                risk_level=risk_level,
                client_updates=client_updates
            )
            qm = server.trust_engine.quarantine_manager
            eligible_cids = [cid for cid in range(1, NUM_CLIENTS + 1) if qm.is_eligible_for_aggregation(cid)]
            surviving_mal = [cid for cid in MALICIOUS_CLIENTS if cid in eligible_cids]
            if surviving_mal:
                el_sizes = [client_sizes[cid - 1] for cid in eligible_cids]
                el_trust = {cid: trust_scores.get(cid, 0.0) for cid in eligible_cids}
                el_weights = server.adaptive_aggregator.calculate_trust_weights(el_sizes, el_trust)
                malicious_weight = sum(w for cid, w in zip(eligible_cids, el_weights) if cid in MALICIOUS_CLIENTS)
            else:
                malicious_weight = 0.0
        else:
            raise ValueError(f"Unknown method: {method_name}")

        server.global_model.load_state_dict(new_parameters)
        loss, accuracy = server.evaluate(test_dataset, return_loss=True)
        last_accuracy = accuracy
        accuracy_percent = accuracy * 100
        round_duration = time.time() - round_start_time

        print(f"Round {round_number}: Acc={accuracy_percent:.2f}% | Loss={loss:.4f} | Agg={selected_aggregator} | MalWeight={malicious_weight:.4f}")

        round_results.append({
            "method": method_name,
            "seed": seed,
            "round": round_number,
            "accuracy": accuracy,
            "accuracy_percent": accuracy_percent,
            "loss": loss,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "threat_type": threat_type,
            "suspicious_clients": suspicious_clients,
            "malicious_weight": malicious_weight,
            "aggregator": selected_aggregator,
            "round_time_sec": round_duration
        })

    return round_results


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def main():
    train_dataset, test_dataset = load_mnist()

    print()
    print("========================================")
    print("REPEATED CONTROLLED COMPARISON (4 METHODS)")
    print("========================================")

    methods = ["STANDARD_FEDAVG", "TRUST_FEDAVG", "TRUST_ROBUST", "TARA-FL"]
    all_results = []
    summary_by_method = {}

    for method in methods:
        method_finals = []
        method_bests = []
        method_losses = []

        for seed in SEEDS:
            res = run_method(method, train_dataset, test_dataset, seed)
            all_results.extend(res)
            method_finals.append(res[-1]["accuracy_percent"])
            method_bests.append(max(r["accuracy_percent"] for r in res))
            method_losses.append(res[-1]["loss"])

        mean_final = statistics.mean(method_finals)
        std_final = statistics.stdev(method_finals) if len(method_finals) > 1 else 0.0
        mean_best = statistics.mean(method_bests)
        std_best = statistics.stdev(method_bests) if len(method_bests) > 1 else 0.0
        mean_loss = statistics.mean(method_losses)

        summary_by_method[method] = {
            "mean_final_accuracy": mean_final,
            "std_final_accuracy": std_final,
            "mean_best_accuracy": mean_best,
            "std_best_accuracy": std_best,
            "mean_final_loss": mean_loss,
            "final_accuracies": method_finals
        }

    # Save CSV
    fieldnames = list(all_results[0].keys())
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Save JSON Summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_by_method, f, indent=2)

    # Print Summary
    print()
    print("========================================")
    print("REPEATED EXPERIMENT SUMMARY")
    print("========================================")
    for method, stats in summary_by_method.items():
        print(f"{method:<20}: Final={stats['mean_final_accuracy']:.2f}% +/- {stats['std_final_accuracy']:.2f}% | Best={stats['mean_best_accuracy']:.2f}% +/- {stats['std_best_accuracy']:.2f}% | Loss={stats['mean_final_loss']:.4f}")
    print("========================================")
    print(f"Results saved to: {RESULT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()