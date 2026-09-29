"""
TARA-FL Publication Visualization Suite.

Generates publication-quality figures from experiment CSV results:
1. Accuracy & Loss Convergence vs. Round across methods and seeds
2. Per-Client Dynamic Trust Score Trajectories (Honest vs. Byzantine)
3. Environmental Round Risk & Dynamic Aggregator Selection Timeline
4. Multi-Seed Method Comparison (Bar & Box plots with error bars)
5. Non-IID vs IID Robustness Breakdown
"""

import os
import csv
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Headless backend for robust rendering and test execution
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_comparison(
        csv_files_dict: Dict[str, str],
        output_path: str = "plots/accuracy_comparison.png",
        title: str = "Federated Learning Accuracy vs Rounds"
):
    """
    Plots multi-seed accuracy curves with standard deviation confidence bands.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for idx, (method_name, file_path) in enumerate(csv_files_dict.items()):
        if not os.path.exists(file_path):
            continue

        rounds = []
        accuracies_by_seed = {}

        with open(file_path, "r", encoding="utf-8") as f:
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

        if not rounds:
            continue

        rounds = sorted(rounds)
        acc_matrix = []
        for seed, r_dict in accuracies_by_seed.items():
            acc_matrix.append([r_dict[r] for r in rounds if r in r_dict])

        acc_matrix = np.array(acc_matrix)
        if acc_matrix.size == 0:
            continue

        mean_acc = np.mean(acc_matrix, axis=0)
        std_acc = np.std(acc_matrix, axis=0)
        color = colors[idx % len(colors)]

        plt.plot(rounds, mean_acc, label=method_name, linewidth=2, marker='o', color=color)
        if len(accuracies_by_seed) > 1:
            plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=color)

    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("Federated Communication Round", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_trust_trajectories(
        csv_path: str,
        num_clients: int = 10,
        malicious_ids: Optional[List[int]] = None,
        output_path: str = "plots/trust_trajectories.png"
):
    """
    Plots per-client dynamic trust score evolution across rounds.
    """
    if malicious_ids is None:
        malicious_ids = [9, 10]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)

    rounds = []
    trust_by_client = {cid: [] for cid in range(1, num_clients + 1)}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
            plt.plot(rounds, trust_by_client[cid], label=f"Client {cid} (Byzantine)", color='red', linestyle='--', linewidth=2.2, marker='x')
        else:
            plt.plot(rounds, trust_by_client[cid], label=f"Client {cid} (Honest)", color='#2ca02c', alpha=0.55, linewidth=1.5)

    plt.axhline(0.75, color='green', linestyle=':', alpha=0.6, label=r'High Trust ($\geq 0.75$)')
    plt.axhline(0.40, color='orange', linestyle=':', alpha=0.6, label=r'Medium Trust ($0.40$)')
    plt.axhline(0.20, color='red', linestyle=':', alpha=0.6, label=r'Quarantine ($\leq 0.20$)')

    plt.title("Client Dynamic Trust Score Trajectories Across Rounds", fontsize=13, fontweight='bold')
    plt.xlabel("Federated Round", fontsize=11)
    plt.ylabel(r"Dynamic Trust Score $T_k^t$", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_risk_and_aggregator(
        csv_path: str,
        output_path: str = "plots/risk_and_aggregator.png"
):
    """
    Plots round risk scores alongside dynamic aggregator selections.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    rounds = []
    risks = []
    aggregators = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "seed" in row and row["seed"] not in ("42", str(reader.fieldnames)):
                continue
            r = int(row["round"])
            if r in rounds:
                continue
            rounds.append(r)
            risks.append(float(row.get("risk_score", 0.0)))
            aggregators.append(row.get("aggregator", "UNKNOWN"))

    ax1.plot(rounds, risks, color='purple', marker='s', linewidth=2, label=r'Environmental Round Risk $\mathcal{R}^t$')
    ax1.axhline(0.30, color='gold', linestyle='--', linewidth=1.5, label='Low/Med Boundary (0.30)')
    ax1.axhline(0.60, color='crimson', linestyle='--', linewidth=1.5, label='Med/High Boundary (0.60)')
    ax1.set_xlabel('Federated Communication Round', fontsize=11)
    ax1.set_ylabel(r'Composite Risk Score $\mathcal{R}^t$', color='purple', fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)

    for i, (r, risk, agg) in enumerate(zip(rounds, risks, aggregators)):
        short_name = agg.replace("TRUST_", "").replace("_", " ")
        ax1.annotate(short_name, (r, risk), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=7.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.35))

    plt.title('Round Risk Evolution and Adaptive Defense Selection', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_method_bar_comparison(
        results_summary: Dict[str, float],
        output_path: str = "plots/method_bar_comparison.png",
        title: str = "Final Accuracy Comparison Under Byzantine Attack"
):
    """
    Bar plot comparing final accuracies across baseline and TARA-FL variants.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)

    methods = list(results_summary.keys())
    accuracies = [results_summary[m] for m in methods]
    colors = ['#e74c3c' if 'FedAvg' in m or 'Baseline' in m else '#2ecc71' for m in methods]

    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black', width=0.55, alpha=0.85)
    plt.ylabel('Final Test Accuracy (%)', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")
