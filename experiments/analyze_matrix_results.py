"""
TARA-FL Rigorous Statistical Analysis and Multi-Metric Synthesis.

Analyzes raw round-by-round trajectory CSVs from the 20-scenario matrix (160 runs).
Calculates:
1. Mean accuracy and standard deviation across seeds.
2. Mean cross-entropy loss.
3. Convergence behavior across rounds.
4. Robustness degradation slopes vs. malicious percentage.
5. Byzantine update influence / weight suppression.
6. Computational / runtime overhead.
7. Trust score distribution & trust zone classification percentages (Safe, Probation, Quarantine).
8. Quarantine trigger frequency and client exclusion counts.
9. RoundRisk score distribution (Low, Medium, High).
10. Aggregation method selection frequency.

Generates comprehensive summary CSV/JSON files and publication-grade figures in plots/analysis/.
"""

import os
import sys
import csv
import json
import glob
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_all_raw_data(raw_dir: str = "experiments/matrix_raw") -> List[Dict[str, Any]]:
    """Loads all per-round rows across all 20 scenario CSV files."""
    files = glob.glob(os.path.join(raw_dir, "raw_scenario_*.csv"))
    all_rows = []
    for fpath in sorted(files):
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if k in ("method", "attack", "data_split", "risk_level", "threat_type", "aggregator"):
                        parsed[k] = v
                    elif k in ("scenario_id", "seed", "round", "suspicious_clients"):
                        parsed[k] = int(v) if v != "N/A" else v
                    elif k.startswith("trust_client_"):
                        parsed[k] = float(v) if v != "N/A" else "N/A"
                    else:
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                all_rows.append(parsed)
    return all_rows


def compute_rigorous_statistics(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes all detailed statistical metrics across experimental dimensions."""
    methods = ["fedavg", "trust_fedavg", "trust_robust", "tara"]
    attacks = ["none", "label_flip", "gradient_scale", "sign_flip"]
    splits = ["iid", "noniid"]
    ratios = [0.0, 0.10, 0.20, 0.30]

    # 1. Performance Summary by (Attack, Ratio, Split, Method)
    performance_table = []
    grouped_runs = defaultdict(lambda: defaultdict(list))

    for r in all_rows:
        key = (r["attack"], r["malicious_ratio"], r["data_split"], r["method"])
        grouped_runs[key][r["seed"]].append(r)

    for (attack, mal_ratio, split, method), seed_dict in grouped_runs.items():
        final_accs = [runs[-1]["accuracy_percent"] for runs in seed_dict.values()]
        best_accs = [max(x["accuracy_percent"] for x in runs) for runs in seed_dict.values()]
        final_losses = [runs[-1]["loss"] for runs in seed_dict.values()]
        mal_weights = [np.mean([x["malicious_weight"] for x in runs]) for runs in seed_dict.values()]
        runtimes = [sum(x["round_time_sec"] for x in runs) for runs in seed_dict.values()]
        mean_risks = [np.mean([x["risk_score"] for x in runs]) for runs in seed_dict.values()]

        # Trust zone counts (for TARA / Trust variants)
        safe_counts, prob_counts, quar_counts = 0, 0, 0
        total_trust_evals = 0
        for runs in seed_dict.values():
            for round_row in runs:
                for cid in range(1, 11):
                    t_val = round_row.get(f"trust_client_{cid}")
                    if isinstance(t_val, (int, float)):
                        total_trust_evals += 1
                        if t_val >= 0.75:
                            safe_counts += 1
                        elif t_val >= 0.40:
                            prob_counts += 1
                        else:
                            quar_counts += 1

        entry = {
            "attack": attack,
            "malicious_ratio": mal_ratio,
            "data_split": split,
            "method": method.upper(),
            "mean_final_accuracy": float(np.mean(final_accs)),
            "std_final_accuracy": float(np.std(final_accs)) if len(final_accs) > 1 else 0.0,
            "mean_best_accuracy": float(np.mean(best_accs)),
            "std_best_accuracy": float(np.std(best_accs)) if len(best_accs) > 1 else 0.0,
            "mean_final_loss": float(np.mean(final_losses)),
            "mean_malicious_weight": float(np.mean(mal_weights)),
            "mean_runtime_sec": float(np.mean(runtimes)),
            "mean_round_risk": float(np.mean(mean_risks)),
            "pct_safe_zone": (safe_counts / total_trust_evals * 100.0) if total_trust_evals > 0 else 100.0,
            "pct_probation_zone": (prob_counts / total_trust_evals * 100.0) if total_trust_evals > 0 else 0.0,
            "pct_quarantine_zone": (quar_counts / total_trust_evals * 100.0) if total_trust_evals > 0 else 0.0,
        }
        performance_table.append(entry)

    # 2. Aggregator Selection Breakdown (TARA-FL specific)
    aggregator_counts = defaultdict(int)
    risk_level_counts = defaultdict(int)
    tara_rows = [r for r in all_rows if r["method"] == "tara"]
    for r in tara_rows:
        aggregator_counts[r["aggregator"]] += 1
        risk_level_counts[r["risk_level"]] += 1

    # 3. Robustness Degradation Slopes (Accuracy loss per 10% malicious increase)
    degradation_slopes = {}
    for split in splits:
        for attack in ["label_flip", "gradient_scale", "sign_flip"]:
            for m in methods:
                m_upper = m.upper()
                acc_0 = next(e["mean_final_accuracy"] for e in performance_table if e["attack"] == "none" and e["data_split"] == split and e["method"] == m_upper)
                acc_30 = next(e["mean_final_accuracy"] for e in performance_table if e["attack"] == attack and abs(e["malicious_ratio"] - 0.30) < 1e-4 and e["data_split"] == split and e["method"] == m_upper)
                drop_30 = acc_0 - acc_30
                degradation_slopes[f"{attack}_{split}_{m_upper}"] = {
                    "clean_accuracy": acc_0,
                    "acc_at_30pct": acc_30,
                    "total_drop_pct": drop_30,
                    "drop_per_10pct_byzantine": drop_30 / 3.0
                }

    # 4. Overall Metric Averages by Method across all Attacked Scenarios
    method_macro_stats = {}
    for m in methods:
        m_upper = m.upper()
        attacked = [e for e in performance_table if e["attack"] != "none" and e["method"] == m_upper]
        clean = [e for e in performance_table if e["attack"] == "none" and e["method"] == m_upper]

        method_macro_stats[m_upper] = {
            "mean_attacked_accuracy": float(np.mean([e["mean_final_accuracy"] for e in attacked])),
            "mean_attacked_loss": float(np.mean([e["mean_final_loss"] for e in attacked])),
            "mean_clean_accuracy": float(np.mean([e["mean_final_accuracy"] for e in clean])),
            "mean_malicious_weight": float(np.mean([e["mean_malicious_weight"] for e in attacked])),
            "mean_runtime_sec": float(np.mean([e["mean_runtime_sec"] for e in attacked])),
        }

    return {
        "performance_table": performance_table,
        "aggregator_selection_counts": dict(aggregator_counts),
        "risk_level_counts": dict(risk_level_counts),
        "degradation_slopes": degradation_slopes,
        "method_macro_stats": method_macro_stats
    }


def generate_rigorous_plots(all_rows: List[Dict[str, Any]], stats: Dict[str, Any], output_dir: str = "plots/analysis"):
    """Generates all 8 publication-grade visualization figures."""
    os.makedirs(output_dir, exist_ok=True)
    methods = ["FEDAVG", "TRUST_FEDAVG", "TRUST_ROBUST", "TARA"]
    colors = {"FEDAVG": "#e74c3c", "TRUST_FEDAVG": "#f39c12", "TRUST_ROBUST": "#3498db", "TARA": "#2ecc71"}
    markers = {"FEDAVG": "x", "TRUST_FEDAVG": "^", "TRUST_ROBUST": "s", "TARA": "o"}

    # -------------------------------------------------------------
    # PLOT 1: Accuracy vs Round Convergence for 4 Key Representative Scenarios
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    key_scenarios = [
        ("none", 0.0, "iid", "Clean Baseline (IID, 0% Byzantine)"),
        ("label_flip", 0.20, "iid", "Label Flip 20% (IID)"),
        ("gradient_scale", 0.20, "iid", "Gradient Scale x10 20% (IID)"),
        ("sign_flip", 0.20, "noniid", "Sign Flip 20% (Non-IID)")
    ]

    for idx, (attack, mal_ratio, split, title) in enumerate(key_scenarios):
        ax = axes[idx // 2, idx % 2]
        for m in ["fedavg", "trust_fedavg", "trust_robust", "tara"]:
            matching = [r for r in all_rows if r["attack"] == attack and abs(r["malicious_ratio"] - mal_ratio) < 1e-4 and r["data_split"] == split and r["method"] == m]
            # Average across seeds per round
            rounds = sorted(list(set(r["round"] for r in matching)))
            accs = [np.mean([r["accuracy_percent"] for r in matching if r["round"] == rnd]) for rnd in rounds]
            ax.plot(rounds, accs, label=m.upper(), color=colors[m.upper()], marker=markers[m.upper()], linewidth=2, markersize=6)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Federated Round", fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_xticks(range(1, 6))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 100)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Convergence Trajectories Across Critical Attack Scenarios", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "1_convergence_curves_key_scenarios.png")
    plt.savefig(p1_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1_path}")

    # -------------------------------------------------------------
    # PLOT 2: Trust Score Distribution (Honest vs Malicious Clients)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5), dpi=300)
    attacks = ["clean", "label_flip", "gradient_scale", "sign_flip"]
    honest_trusts = []
    malicious_trusts = []

    for att in ["none", "label_flip", "gradient_scale", "sign_flip"]:
        att_rows = [r for r in all_rows if r["attack"] == att and r["method"] == "tara" and r["round"] == 5]
        h_vals = []
        m_vals = []
        for r in att_rows:
            mal_count = int(10 * r["malicious_ratio"])
            mal_ids = set(range(10 - mal_count + 1, 11))
            for cid in range(1, 11):
                t_val = r.get(f"trust_client_{cid}")
                if isinstance(t_val, (int, float)):
                    if cid in mal_ids:
                        m_vals.append(t_val)
                    else:
                        h_vals.append(t_val)
        honest_trusts.append(np.mean(h_vals) if h_vals else 1.0)
        malicious_trusts.append(np.mean(m_vals) if m_vals else 0.0)

    x = np.arange(len(attacks))
    width = 0.35
    plt.bar(x - width/2, honest_trusts, width, label="Honest Clients", color="#2ecc71", edgecolor="black", alpha=0.85)
    plt.bar(x + width/2, malicious_trusts, width, label="Malicious Clients", color="#e74c3c", edgecolor="black", alpha=0.85)
    plt.axhline(0.75, color="gray", linestyle="--", label="Safe Boundary (0.75)")
    plt.axhline(0.40, color="crimson", linestyle=":", label="Quarantine Boundary (0.40)")
    plt.xticks(x, [a.replace("_", " ").title() for a in attacks], fontsize=10)
    plt.ylabel("Mean Final Trust Score", fontsize=11)
    plt.title("Dynamic Trust Score Separation: Honest vs. Malicious Clients", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "2_trust_score_distribution.png")
    plt.savefig(p2_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2_path}")

    # -------------------------------------------------------------
    # PLOT 3: Trust Zone Percentage Breakdown
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    tara_table = [e for e in stats["performance_table"] if e["method"] == "TARA" and e["attack"] != "none"]
    attack_types = ["label_flip", "gradient_scale", "sign_flip"]
    safe_pcts = [np.mean([e["pct_safe_zone"] for e in tara_table if e["attack"] == a]) for a in attack_types]
    prob_pcts = [np.mean([e["pct_probation_zone"] for e in tara_table if e["attack"] == a]) for a in attack_types]
    quar_pcts = [np.mean([e["pct_quarantine_zone"] for e in tara_table if e["attack"] == a]) for a in attack_types]

    x = np.arange(len(attack_types))
    plt.bar(x, safe_pcts, label="Safe Zone (≥0.75)", color="#2ecc71", edgecolor="black", width=0.45)
    plt.bar(x, prob_pcts, bottom=safe_pcts, label="Probation Zone [0.40, 0.75)", color="#f39c12", edgecolor="black", width=0.45)
    bottom_quar = np.array(safe_pcts) + np.array(prob_pcts)
    plt.bar(x, quar_pcts, bottom=bottom_quar, label="Quarantine Zone (<0.40)", color="#e74c3c", edgecolor="black", width=0.45)

    plt.xticks(x, [a.replace("_", " ").title() for a in attack_types], fontsize=10)
    plt.ylabel("Evaluation Percentage (%)", fontsize=11)
    plt.title("Client Trust Zone Distribution Across Attack Types (TARA-FL)", fontsize=12, fontweight="bold")
    plt.ylim(0, 105)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3_path = os.path.join(output_dir, "3_trust_zone_breakdown.png")
    plt.savefig(p3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3_path}")

    # -------------------------------------------------------------
    # PLOT 4: RoundRisk Score Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    tara_evals = [r for r in all_rows if r["method"] == "tara"]
    clean_risks = [r["risk_score"] for r in tara_evals if r["attack"] == "none"]
    lf_risks = [r["risk_score"] for r in tara_evals if r["attack"] == "label_flip"]
    gs_risks = [r["risk_score"] for r in tara_evals if r["attack"] == "gradient_scale"]
    sf_risks = [r["risk_score"] for r in tara_evals if r["attack"] == "sign_flip"]

    risk_data = [clean_risks, lf_risks, gs_risks, sf_risks]
    box = plt.boxplot(risk_data, patch_artist=True)
    plt.xticks([1, 2, 3, 4], ["Clean", "Label Flip", "Gradient Scale", "Sign Flip"], fontsize=10)
    box_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.axhline(0.30, color="orange", linestyle="--", label="Low/Medium Risk Threshold (0.30)")
    plt.axhline(0.60, color="crimson", linestyle=":", label="Medium/High Risk Threshold (0.60)")
    plt.ylabel("Composite Round Risk Score", fontsize=11)
    plt.title("Composite Round-Risk Score Distribution by Attack Category", fontsize=12, fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p4_path = os.path.join(output_dir, "4_round_risk_distribution.png")
    plt.savefig(p4_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p4_path}")

    # -------------------------------------------------------------
    # PLOT 5: Aggregation Selection Frequency
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=300)
    sel_counts = stats["aggregator_selection_counts"]
    clean_labels = [k.replace("TRUST_", "").replace("_", " ") for k in sel_counts.keys()]
    plt.pie(list(sel_counts.values()), labels=clean_labels, autopct='%1.1f%%',
            startangle=140, colors=["#3498db", "#2ecc71", "#e74c3c"],
            wedgeprops={"edgecolor": "black", "linewidth": 1.2})
    plt.title("Adaptive Aggregator Selection Frequency in TARA-FL", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p5_path = os.path.join(output_dir, "5_aggregation_selection_frequency.png")
    plt.savefig(p5_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p5_path}")

    # -------------------------------------------------------------
    # PLOT 6: Robustness Degradation vs Malicious % (IID & Non-IID)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)
    for ax, split, title in [(ax1, "iid", "IID Data Distribution"), (ax2, "noniid", "Non-IID (α=0.5) Distribution")]:
        ratios = [0.10, 0.20, 0.30]
        for m in methods:
            accs = []
            for r in ratios:
                # Average across all 3 attacks for this ratio and split
                match_entries = [e for e in stats["performance_table"] if e["data_split"] == split and abs(e["malicious_ratio"] - r) < 1e-4 and e["method"] == m and e["attack"] != "none"]
                accs.append(np.mean([e["mean_final_accuracy"] for e in match_entries]))
            ax.plot([r * 100 for r in ratios], accs, label=m, color=colors[m], marker=markers[m], linewidth=2.2, markersize=7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Byzantine Client Proportion (%)", fontsize=10)
        ax.set_xticks([10, 20, 30])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(40, 95)
        if split == "iid":
            ax.set_ylabel("Macro Mean Test Accuracy (%)", fontsize=10)
            ax.legend(loc="lower left", fontsize=9)

    plt.suptitle("Robustness Degradation Across Byzantine Proportions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p6_path = os.path.join(output_dir, "6_robustness_degradation_curves.png")
    plt.savefig(p6_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p6_path}")

    # -------------------------------------------------------------
    # PLOT 7: Computational / Runtime Overhead
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    macro_runtimes = [stats["method_macro_stats"][m]["mean_runtime_sec"] for m in methods]
    bars = plt.bar(methods, macro_runtimes, color=[colors[m] for m in methods], edgecolor="black", width=0.5, alpha=0.88)
    plt.ylabel("Mean 5-Round Wall-Clock Runtime (s)", fontsize=11)
    plt.title("Computational Overhead Across Federated Aggregation Frameworks", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(0, max(macro_runtimes) * 1.25)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p7_path = os.path.join(output_dir, "7_runtime_overhead_comparison.png")
    plt.savefig(p7_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p7_path}")

    # -------------------------------------------------------------
    # PLOT 8: Byzantine Weight Suppression Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    macro_weights = [stats["method_macro_stats"][m]["mean_malicious_weight"] for m in methods]
    bars = plt.bar(methods, macro_weights, color=[colors[m] for m in methods], edgecolor="black", width=0.5, alpha=0.88)
    plt.ylabel("Effective Byzantine Weight in Global Model", fontsize=11)
    plt.title("Effective Byzantine Influence Suppression in Aggregation", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(0, 0.25)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.4f} ({h*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p8_path = os.path.join(output_dir, "8_byzantine_weight_suppression.png")
    plt.savefig(p8_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p8_path}")


def main():
    print("================================================================")
    print("      TARA-FL RIGOROUS EXPERIMENTAL ANALYSIS & SYNTHESIS        ")
    print("================================================================")

    all_rows = load_all_raw_data()
    print(f"Loaded {len(all_rows)} total round-level measurement records.")

    stats = compute_rigorous_statistics(all_rows)

    # 1. Save Structured Summary CSV
    csv_out = "experiments/rigorous_analysis_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(stats["performance_table"][0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats["performance_table"])
    print(f"Saved summary CSV: {csv_out}")

    # 2. Save Structured Summary JSON
    json_out = "experiments/rigorous_analysis_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved summary JSON: {json_out}")

    # 3. Generate Visualizations
    generate_rigorous_plots(all_rows, stats, output_dir="plots/analysis")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
