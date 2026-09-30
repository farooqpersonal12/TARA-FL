"""
TARA-FL Unified Controlled Experiment Runner.

Executes controlled, reproducible federated learning experiments across four methods:
1. Standard FedAvg (fedavg): Baseline aggregation without defense or trust weighting.
2. Trust + FedAvg (trust_fedavg): Dynamic continuous trust analysis with Trust-Aware FedAvg.
3. Trust + Fixed Robust (trust_robust): Dynamic trust analysis with fixed Robust Trimming.
4. Full TARA-FL (tara): Continuous trust, QuarantineManager, multi-factor RoundRisk, and Adaptive Aggregation.
5. All (all): Executes all four methods sequentially under identical conditions.

Usage examples:
    # Run controlled 4-way comparison under 20% label-flip attack (10 clients, 5 rounds)
    python experiments/run_unified.py --method all --num_clients 10 --malicious_ratio 0.2 --attack label_flip --rounds 5 --seeds 42

    # Run single method
    python experiments/run_unified.py --method tara --num_clients 10 --malicious_ratio 0.2 --attack label_flip --rounds 5 --seeds 42
"""

import argparse
import csv
import json
import os
import sys
import random
import time
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from visualization.plots import (
    plot_accuracy_comparison,
    plot_loss_comparison,
    plot_trust_trajectories,
    plot_risk_and_aggregator,
    plot_method_bar_comparison
)
from visualization.report_generator import ReportGenerator, generate_experiment_report


# ==========================================================
# STANDARD FEDAVG (BASELINE A HELPER)
# ==========================================================

def fedavg(client_parameters: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
    """Standard FedAvg — weighted average by dataset size."""
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
# RUN SINGLE EXPERIMENT
# ==========================================================

def run_experiment(config, seed: int, method_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run one complete federated learning experiment with the given configuration and seed.

    Returns a list of per-round result dictionaries.
    """
    method = method_override if method_override is not None else config.method
    num_clients = config.num_clients
    num_rounds = config.rounds
    local_epochs = config.local_epochs

    # ======================================================
    # REPRODUCIBILITY SEEDING
    # ======================================================
    random.seed(seed)
    torch.manual_seed(seed)

    print()
    print("########################################")
    print(f"METHOD: {method.upper()} | Seed: {seed} | {num_clients} clients | {num_rounds} rounds")
    print("########################################")

    # ======================================================
    # CREATE SERVER FOR THE METHOD
    # ======================================================
    enable_quarantine = (method == "tara")
    server = Server(
        detector_type=config.detector_type,
        enable_quarantine=enable_quarantine
    )

    # ======================================================
    # LOAD DATASET & CREATE PARTITIONS
    # ======================================================
    train_dataset, test_dataset = load_mnist()

    if config.data_split == "noniid":
        client_datasets = create_clients_noniid(
            train_dataset,
            num_clients=num_clients,
            alpha=config.alpha,
            seed=seed
        )
    else:
        client_datasets = create_clients(
            train_dataset,
            num_clients=num_clients
        )

    # ======================================================
    # DETERMINE MALICIOUS CLIENTS
    # ======================================================
    num_malicious = int(num_clients * config.malicious_ratio)
    malicious_ids = list(range(num_clients - num_malicious + 1, num_clients + 1)) if num_malicious > 0 else []

    # ======================================================
    # CREATE CLIENTS & CONFIGURE ATTACKS
    # ======================================================
    clients = []
    gradient_attacks = {}

    for i in range(num_clients):
        client_id = i + 1
        client_dataset = client_datasets[i]

        # Apply label-flipping attack
        if client_id in malicious_ids and config.attack == "label_flip":
            client_dataset = LabelFlipDataset(
                client_dataset,
                flip_ratio=config.flip_ratio,
                seed=seed
            )

        # Prepare gradient attack
        if client_id in malicious_ids and config.attack == "gradient_scale":
            gradient_attacks[client_id] = GradientScaleAttack(scale_factor=config.scale_factor)

        if client_id in malicious_ids and config.attack == "sign_flip":
            gradient_attacks[client_id] = SignFlipAttack()

        client = Client(client_id=client_id, dataset=client_dataset)
        clients.append(client)

    # ======================================================
    # FEDERATED TRAINING LOOP
    # ======================================================
    results = []
    last_accuracy = None

    for round_number in range(1, num_rounds + 1):
        round_start_time = time.time()

        print()
        print(f"--- Federated Round {round_number}/{num_rounds} [{method.upper()}] ---")

        # Global parameters
        global_parameters = server.global_model.state_dict()

        # Local training
        client_parameters = []
        client_sizes = []
        client_updates = {}

        for client in clients:
            client.set_model(global_parameters)
            client.train(epochs=local_epochs)
            parameters = client.get_parameters()
            update = client.get_update(global_parameters)

            # Apply gradient poisoning if active
            if client.client_id in gradient_attacks:
                update = gradient_attacks[client.client_id].apply(update)
                poisoned_params = {}
                for k in global_parameters:
                    poisoned_params[k] = global_parameters[k] + update[k]
                parameters = poisoned_params

            client_parameters.append(parameters)
            client_sizes.append(len(client.dataset))
            client_updates[client.client_id] = update

        # ==================================================
        # METHOD-SPECIFIC AGGREGATION PIPELINE
        # ==================================================

        if method == "fedavg":
            # Baseline 1: Standard FedAvg
            new_parameters = fedavg(client_parameters, client_sizes)
            selected_aggregator = "STANDARD_FEDAVG"
            risk_score = 0.0
            risk_level = "N/A"
            threat_type = "N/A"
            suspicious_clients = 0
            trust_scores = {}

            # Malicious influence calculation
            total_samples = sum(client_sizes)
            malicious_weight = (
                sum(client_sizes[cid - 1] for cid in malicious_ids) / total_samples
                if total_samples > 0 and malicious_ids else 0.0
            )

        elif method == "trust_fedavg":
            # Baseline 2: Trust + FedAvg (Trust-Aware FedAvg)
            distances, pid_scores = server.detector.calculate_scores(client_updates)
            all_pid_scores = list(pid_scores.values())

            trust_scores = {}
            for cid, score in pid_scores.items():
                res = server.trust_engine.calculate_trust(cid, score, all_pid_scores)
                trust_scores[cid] = res["trust"]

            client_ids_list = list(range(1, num_clients + 1))
            new_parameters = server.adaptive_aggregator.strategy_fedavg.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_ids=client_ids_list
            )
            selected_aggregator = "TRUST_AWARE_FEDAVG"
            risk_score = 0.0
            risk_level = "N/A"
            threat_type = "N/A"
            suspicious_clients = sum(1 for t in trust_scores.values() if t < 0.75)

            weights = server.adaptive_aggregator.strategy_fedavg.calculate_trust_weights(client_sizes, trust_scores)
            malicious_weight = sum(weights[cid - 1] for cid in malicious_ids) if malicious_ids and len(weights) == num_clients else 0.0

        elif method == "trust_robust":
            # Baseline 3: Trust + Fixed Robust Aggregation (Robust Trimming)
            distances, pid_scores = server.detector.calculate_scores(client_updates)
            all_pid_scores = list(pid_scores.values())

            trust_scores = {}
            for cid, score in pid_scores.items():
                res = server.trust_engine.calculate_trust(cid, score, all_pid_scores)
                trust_scores[cid] = res["trust"]

            client_ids_list = list(range(1, num_clients + 1))
            new_parameters = server.adaptive_aggregator.strategy_robust.aggregate(
                client_parameters=client_parameters,
                client_sizes=client_sizes,
                trust_scores=trust_scores,
                client_updates=client_updates,
                risk_level="MEDIUM",
                client_ids=client_ids_list
            )
            selected_aggregator = "TRUST_WEIGHTED_ROBUST"
            risk_score = 0.0
            risk_level = "N/A"
            threat_type = "N/A"
            suspicious_clients = sum(1 for t in trust_scores.values() if t < 0.75)

            # Compute effective weights after trimming
            distances_map = server.adaptive_aggregator.calculate_update_distances(client_updates) if client_updates else {}
            surviving = server.adaptive_aggregator.select_robust_clients(client_ids_list, distances_map, "MEDIUM")
            surviving_sizes = [client_sizes[cid - 1] for cid in surviving]
            surviving_trust = {cid: trust_scores.get(cid, 0.0) for cid in surviving}
            surv_weights = server.adaptive_aggregator.calculate_trust_weights(surviving_sizes, surviving_trust)
            malicious_weight = sum(w for cid, w in zip(surviving, surv_weights) if cid in malicious_ids)

        elif method == "tara":
            # Method 4: Full TARA-FL Pipeline (Trust + Quarantine + RoundRisk + Adaptive Aggregation)
            distances, pid_scores = server.detector.calculate_scores(client_updates)
            all_pid_scores = list(pid_scores.values())

            trust_scores = {}
            for cid, score in pid_scores.items():
                res = server.trust_engine.calculate_trust(cid, score, all_pid_scores)
                trust_scores[cid] = res["trust"]

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

            # Malicious influence: quarantined clients contribute 0 weight
            qm = server.trust_engine.quarantine_manager
            eligible_cids = [cid for cid in range(1, num_clients + 1) if qm.is_eligible_for_aggregation(cid)]
            surviving_malicious = [cid for cid in malicious_ids if cid in eligible_cids]
            if surviving_malicious:
                eligible_sizes = [client_sizes[cid - 1] for cid in eligible_cids]
                eligible_trust = {cid: trust_scores.get(cid, 0.0) for cid in eligible_cids}
                el_weights = server.adaptive_aggregator.calculate_trust_weights(eligible_sizes, eligible_trust)
                malicious_weight = sum(w for cid, w in zip(eligible_cids, el_weights) if cid in malicious_ids)
            else:
                malicious_weight = 0.0

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # ==================================================
        # UPDATE & EVALUATE GLOBAL MODEL
        # ==================================================
        server.global_model.load_state_dict(new_parameters)
        loss, accuracy = server.evaluate(test_dataset, return_loss=True)
        last_accuracy = accuracy
        accuracy_percent = accuracy * 100
        round_duration = time.time() - round_start_time

        print(f"Round {round_number} Result: Accuracy={accuracy_percent:.2f}% | Loss={loss:.4f} | Aggregator={selected_aggregator} | Malicious Weight={malicious_weight:.4f} | Time={round_duration:.2f}s")

        # ==================================================
        # STORE ROUND METRICS
        # ==================================================
        row = {
            "method": method,
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
        }

        # Per-client trust tracking
        for cid in range(1, num_clients + 1):
            row[f"trust_client_{cid}"] = trust_scores.get(cid, "N/A")

        results.append(row)

    return results


# ==========================================================
# ARGUMENT PARSER
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TARA-FL Unified Controlled Experiment Runner"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["tara", "fedavg", "trust_fedavg", "trust_robust", "all"],
        help="Aggregation method to run ('all' runs all 4 methods in a controlled comparison)"
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="Total number of federated clients"
    )

    parser.add_argument(
        "--malicious_ratio",
        type=float,
        default=0.2,
        help="Fraction of Byzantine / malicious clients"
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="label_flip",
        choices=["none", "label_flip", "gradient_scale", "sign_flip"],
        help="Byzantine attack type"
    )

    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=0.75,
        help="Label flip corruption ratio"
    )

    parser.add_argument(
        "--scale_factor",
        type=float,
        default=10.0,
        help="Scale factor for gradient scaling attack"
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
        help="Dirichlet concentration parameter for non-IID split"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated communication rounds"
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local client training epochs per round"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for repeated experimental runs"
    )

    parser.add_argument(
        "--detector_type",
        type=str,
        default="standard",
        choices=["standard", "robust"],
        help="PID anomaly detector type"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Directory to save experimental results"
    )

    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Directory to save generated comparison plots"
    )

    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save generated comparison reports"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment identifier name"
    )

    return parser.parse_args()


# ==========================================================
# MAIN EXECUTION ROUTINE
# ==========================================================

def main():
    config = parse_args()

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    os.makedirs(config.reports_dir, exist_ok=True)

    num_malicious = int(config.num_clients * config.malicious_ratio)
    malicious_ids = list(range(config.num_clients - num_malicious + 1, config.num_clients + 1)) if num_malicious > 0 else []
    scenario_tag = config.experiment_name or f"{config.num_clients}clients_{num_malicious}mal_{config.attack}_{config.data_split}"

    print()
    print("=================================================================")
    print("        TARA-FL CONTROLLED EXPERIMENTAL BENCHMARK RUNNER         ")
    print("=================================================================")
    print(f"Scenario:          {scenario_tag}")
    print(f"Target Method(s):  {config.method}")
    print(f"Total Clients:     {config.num_clients} (Malicious: {num_malicious})")
    print(f"Attack Type:       {config.attack} (Flip Ratio: {config.flip_ratio})")
    print(f"Data Split:        {config.data_split} (Alpha: {config.alpha})")
    print(f"Rounds:            {config.rounds}")
    print(f"Seeds:             {config.seeds}")
    print("=================================================================")

    methods_to_run = ["fedavg", "trust_fedavg", "trust_robust", "tara"] if config.method == "all" else [config.method]

    all_method_results: Dict[str, List[Dict[str, Any]]] = {}
    all_method_summaries: Dict[str, Dict[str, Any]] = {}
    csv_file_map: Dict[str, str] = {}

    benchmark_start_time = time.time()

    for method in methods_to_run:
        print()
        print(f"===============================================================")
        print(f"  EXECUTING METHOD: {method.upper()}")
        print(f"===============================================================")

        method_all_rounds = []
        method_seed_summaries = []

        method_file_name = f"results_{method}_{scenario_tag}.csv"
        method_csv_path = os.path.join(config.output_dir, method_file_name)
        csv_file_map[method.upper()] = method_csv_path

        for seed in config.seeds:
            seed_results = run_experiment(config, seed, method_override=method)
            method_all_rounds.extend(seed_results)

            final_acc = seed_results[-1]["accuracy_percent"]
            best_acc = max(r["accuracy_percent"] for r in seed_results)
            final_loss = seed_results[-1]["loss"]
            min_loss = min(r["loss"] for r in seed_results)
            mean_mal_weight = sum(r["malicious_weight"] for r in seed_results) / len(seed_results)
            total_seed_time = sum(r["round_time_sec"] for r in seed_results)

            method_seed_summaries.append({
                "seed": seed,
                "final_accuracy": final_acc,
                "best_accuracy": best_acc,
                "final_loss": final_loss,
                "min_loss": min_loss,
                "mean_malicious_weight": mean_mal_weight,
                "total_time_sec": total_seed_time
            })

        all_method_results[method.upper()] = method_all_rounds

        # Save per-method round results CSV
        if method_all_rounds:
            fieldnames = list(method_all_rounds[0].keys())
            with open(method_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(method_all_rounds)

        # Compute summary metrics across seeds
        final_accs = [s["final_accuracy"] for s in method_seed_summaries]
        best_accs = [s["best_accuracy"] for s in method_seed_summaries]
        final_losses = [s["final_loss"] for s in method_seed_summaries]
        mean_times = [s["total_time_sec"] for s in method_seed_summaries]

        mean_final_acc = float(sum(final_accs) / len(final_accs))
        std_final_acc = float((sum((x - mean_final_acc) ** 2 for x in final_accs) / max(1, len(final_accs) - 1)) ** 0.5) if len(final_accs) > 1 else 0.0

        mean_best_acc = float(sum(best_accs) / len(best_accs))
        std_best_acc = float((sum((x - mean_best_acc) ** 2 for x in best_accs) / max(1, len(best_accs) - 1)) ** 0.5) if len(best_accs) > 1 else 0.0

        mean_final_loss = float(sum(final_losses) / len(final_losses))
        mean_runtime = float(sum(mean_times) / len(mean_times))

        summary_dict = {
            "method": method.upper(),
            "mean_final_accuracy": mean_final_acc,
            "std_final_accuracy": std_final_acc,
            "mean_best_accuracy": mean_best_acc,
            "std_best_accuracy": std_best_acc,
            "mean_final_loss": mean_final_loss,
            "mean_runtime_sec": mean_runtime,
            "per_seed_runs": method_seed_summaries
        }
        all_method_summaries[method.upper()] = summary_dict

        # Save per-method summary JSON
        summary_json_path = os.path.join(config.output_dir, f"summary_{method}_{scenario_tag}.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

    total_benchmark_time = time.time() - benchmark_start_time

    # ======================================================
    # MULTI-METHOD COMPARISON OUTPUT & VISUALIZATION
    # ======================================================
    if len(methods_to_run) > 1:
        # 1. Save combined comparison summary CSV
        comp_summary_path = os.path.join(config.output_dir, f"summary_comparison_{scenario_tag}.csv")
        with open(comp_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "method", "mean_final_accuracy", "std_final_accuracy",
                "mean_best_accuracy", "std_best_accuracy", "mean_final_loss", "mean_runtime_sec"
            ])
            writer.writeheader()
            for m_key, s in all_method_summaries.items():
                writer.writerow({
                    "method": m_key,
                    "mean_final_accuracy": f"{s['mean_final_accuracy']:.2f}%",
                    "std_final_accuracy": f"{s['std_final_accuracy']:.2f}%",
                    "mean_best_accuracy": f"{s['mean_best_accuracy']:.2f}%",
                    "std_best_accuracy": f"{s['std_best_accuracy']:.2f}%",
                    "mean_final_loss": f"{s['mean_final_loss']:.4f}",
                    "mean_runtime_sec": f"{s['mean_runtime_sec']:.2f}s"
                })

        # 2. Save combined comparison JSON
        comp_json_path = os.path.join(config.output_dir, f"summary_comparison_{scenario_tag}.json")
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(all_method_summaries, f, indent=2)

        # 3. Generate Comparative Plots
        acc_plot_path = os.path.join(config.plots_dir, f"accuracy_comparison_{scenario_tag}.png")
        loss_plot_path = os.path.join(config.plots_dir, f"loss_comparison_{scenario_tag}.png")
        bar_plot_path = os.path.join(config.plots_dir, f"method_bar_comparison_{scenario_tag}.png")

        plot_accuracy_comparison(
            csv_files_dict=csv_file_map,
            output_path=acc_plot_path,
            title=f"Test Accuracy vs Rounds ({scenario_tag})"
        )

        plot_loss_comparison(
            csv_files_dict=csv_file_map,
            output_path=loss_plot_path,
            title=f"Test Loss vs Rounds ({scenario_tag})"
        )

        final_acc_summary = {m: all_method_summaries[m]["mean_final_accuracy"] for m in all_method_summaries}
        plot_method_bar_comparison(
            results_summary=final_acc_summary,
            output_path=bar_plot_path,
            title=f"Final Accuracy Across Methods ({scenario_tag})"
        )

        # Trust and Risk plots for TARA-FL
        if "TARA" in csv_file_map and os.path.exists(csv_file_map["TARA"]):
            tara_csv = csv_file_map["TARA"]
            plot_trust_trajectories(
                csv_path=tara_csv,
                num_clients=config.num_clients,
                malicious_ids=malicious_ids,
                output_path=os.path.join(config.plots_dir, f"trust_trajectories_{scenario_tag}.png")
            )
            plot_risk_and_aggregator(
                csv_path=tara_csv,
                output_path=os.path.join(config.plots_dir, f"risk_and_aggregator_{scenario_tag}.png")
            )

        # 4. Generate Comprehensive Markdown and HTML Reports
        attack_info = {
            "type": config.attack,
            "byzantine_ratio": f"{int(config.malicious_ratio * 100)}%",
            "dataset": "MNIST",
            "num_clients": config.num_clients,
            "partition": config.data_split
        }
        generate_experiment_report(
            experiment_name=scenario_tag,
            baselines_data=all_method_results,
            attack_info=attack_info,
            output_dir=config.reports_dir
        )

    # ======================================================
    # FINAL DISPLAY
    # ======================================================
    print()
    print("=================================================================")
    print("                    FINAL BENCHMARK SUMMARY                      ")
    print("=================================================================")
    print(f"{'Method':<25} | {'Final Acc (%)':<15} | {'Best Acc (%)':<15} | {'Final Loss':<12} | {'Time (s)':<10}")
    print("-----------------------------------------------------------------")
    for m_key, s in all_method_summaries.items():
        print(f"{m_key:<25} | {s['mean_final_accuracy']:6.2f}% +/- {s['std_final_accuracy']:4.2f}% | {s['mean_best_accuracy']:6.2f}% +/- {s['std_best_accuracy']:4.2f}% | {s['mean_final_loss']:10.4f} | {s['mean_runtime_sec']:8.2f}s")
    print("=================================================================")
    print(f"Total Benchmark Execution Time: {total_benchmark_time:.2f}s")


if __name__ == "__main__":
    main()
