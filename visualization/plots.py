"""
TARA-FL Visualization Module.

Generates publication-quality plots from experiment CSV results:
1. Accuracy vs. Round across methods and seeds
2. Trust Score Trajectories across rounds (Honest vs. Malicious)
3. Round Risk & Dynamic Aggregator Selection Timeline
4. Multi-Seed Performance Comparison (Bar / Box plots)
"""

import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_comparison(csv_files_dict, output_path="plots/accuracy_comparison.png", title="Federated Learning Accuracy vs Rounds"):
    """
    Plots multi-seed accuracy curves with standard deviation shading.

    Parameters:
        csv_files_dict: dict of method_name -> list of round dicts or csv path
        output_path: path to save output figure
        title: plot title
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)

    for method_name, file_path in csv_files_dict.items():
        # Load CSV data
        rounds = []
        accuracies_by_seed = {}

        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seed = row.get("seed", "default")
                r = int(row["round"])
                acc = float(row.get("accuracy_percent", float(row.get("accuracy", 0.0)) * 100))
                
                if seed not in accuracies_by_seed:
                    accuracies_by_seed[seed] = {}
                accuracies_by_seed[seed][r] = acc
                if r not in rounds:
                    rounds.append(r)

        rounds = sorted(rounds)
        acc_matrix = []
        for seed, r_dict in accuracies_by_seed.items():
            acc_matrix.append([r_dict[r] for r in rounds])

        acc_matrix = np.array(acc_matrix)
        mean_acc = np.mean(acc_matrix, axis=0)
        std_acc = np.std(acc_matrix, axis=0)

        plt.plot(rounds, mean_acc, label=method_name, linewidth=2, marker='o')
        if len(accuracies_by_seed) > 1:
            plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15)

    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("Federated Round", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_trust_trajectories(csv_path, num_clients=10, malicious_ids=None, output_path="plots/trust_trajectories.png"):
    """
    Plots per-client trust score evolution across rounds.
    """
    if malicious_ids is None:
        malicious_ids = [9, 10]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)

    rounds = []
    trust_by_client = {cid: [] for cid in range(1, num_clients + 1)}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # If multi-seed, filter for first seed or seed 42
            if "seed" in row and row["seed"] not in ("42", str(reader.fieldnames)):
                continue
            r = int(row["round"])
            if r in rounds:
                continue
            rounds.append(r)
            for cid in range(1, num_clients + 1):
                key = f"trust_client_{cid}"
                if key in row and row[key] != "N/A":
                    trust_by_client[cid].append(float(row[key]))

    for cid in range(1, num_clients + 1):
        if not trust_by_client[cid]:
            continue
        if cid in malicious_ids:
            plt.plot(rounds, trust_by_client[cid], label=f"Client {cid} (Malicious)", color='red', linestyle='--', linewidth=2, marker='x')
        else:
            plt.plot(rounds, trust_by_client[cid], label=f"Client {cid} (Honest)", alpha=0.6, linewidth=1.5)

    plt.title("Client Trust Score Trajectory Across Federated Rounds", fontsize=13, fontweight='bold')
    plt.xlabel("Federated Round", fontsize=11)
    plt.ylabel("Trust Score [0.0 - 1.0]", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_risk_and_aggregator(csv_path, output_path="plots/risk_and_aggregator.png"):
    """
    Plots round risk scores alongside dynamic aggregator selections.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    rounds = []
    risks = []
    aggregators = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "seed" in row and row["seed"] not in ("42", str(reader.fieldnames)):
                continue
            r = int(row["round"])
            if r in rounds:
                continue
            rounds.append(r)
            risks.append(float(row["risk_score"]))
            aggregators.append(row.get("aggregator", "UNKNOWN"))

    ax1.plot(rounds, risks, color='purple', marker='s', linewidth=2, label='Round Risk Score')
    ax1.axhline(0.30, color='gold', linestyle=':', label='Low/Medium Threshold (0.30)')
    ax1.axhline(0.60, color='crimson', linestyle=':', label='Medium/High Threshold (0.60)')
    ax1.set_xlabel('Federated Round', fontsize=11)
    ax1.set_ylabel('Risk Score [0.0 - 1.0]', color='purple', fontsize=11)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Annotate aggregator selections
    for i, (r, risk, agg) in enumerate(zip(rounds, risks, aggregators)):
        ax1.annotate(agg, (r, risk), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))

    plt.title('Round Risk Evolution and Dynamic Defense Strategy Selection', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    print("Visualization module loaded. Can be imported or run with result files.")
