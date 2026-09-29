"""
Unit Tests for TARA-FL Visualization, Dashboard & Reporting Suite.
"""

import os
import csv
import json
import tempfile
import urllib.request
from typing import Dict, Any

import pytest
from visualization.plots import (
    plot_accuracy_comparison,
    plot_trust_trajectories,
    plot_risk_and_aggregator,
    plot_method_bar_comparison
)
from visualization.report_generator import ReportGenerator, generate_experiment_report
from visualization.dashboard import DashboardRequestHandler, start_dashboard_server


@pytest.fixture
def sample_csv_data(tmp_path):
    """Creates temporary sample CSV files mimicking federated learning runs."""
    fedavg_csv = tmp_path / "baseline_a_fedavg.csv"
    tara_csv = tmp_path / "baseline_d_tara.csv"

    # Baseline A CSV
    with open(fedavg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "seed", "accuracy_percent", "loss", "risk_score", "aggregator", "trust_client_1", "trust_client_2"])
        writer.writeheader()
        for r in range(1, 6):
            writer.writerow({
                "round": r,
                "seed": "42",
                "accuracy_percent": 50.0 + r * 5,
                "loss": 1.5 - r * 0.2,
                "risk_score": 0.1,
                "aggregator": "FEDAVG",
                "trust_client_1": 1.0,
                "trust_client_2": 0.9
            })

    # Baseline D CSV
    with open(tara_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "seed", "accuracy_percent", "loss", "risk_score", "aggregator", "trust_client_1", "trust_client_2"])
        writer.writeheader()
        for r in range(1, 6):
            writer.writerow({
                "round": r,
                "seed": "42",
                "accuracy_percent": 60.0 + r * 6.5,
                "loss": 1.2 - r * 0.2,
                "risk_score": 0.45 if r > 2 else 0.15,
                "aggregator": "TRUST_WEIGHTED_MEDIAN" if r > 2 else "TRUST_FEDAVG",
                "trust_client_1": 0.95,
                "trust_client_2": 0.15 if r > 2 else 0.8
            })

    return {
        "fedavg": str(fedavg_csv),
        "tara": str(tara_csv),
        "dir": str(tmp_path)
    }


def test_plot_accuracy_comparison(sample_csv_data, tmp_path):
    out_img = str(tmp_path / "accuracy.png")
    csv_dict = {
        "Baseline A (FedAvg)": sample_csv_data["fedavg"],
        "Baseline D (TARA-FL)": sample_csv_data["tara"]
    }
    plot_accuracy_comparison(csv_dict, output_path=out_img)
    assert os.path.exists(out_img)
    assert os.path.getsize(out_img) > 1000


def test_plot_trust_trajectories(sample_csv_data, tmp_path):
    out_img = str(tmp_path / "trust.png")
    plot_trust_trajectories(
        csv_path=sample_csv_data["tara"],
        num_clients=2,
        malicious_ids=[2],
        output_path=out_img
    )
    assert os.path.exists(out_img)
    assert os.path.getsize(out_img) > 1000


def test_plot_risk_and_aggregator(sample_csv_data, tmp_path):
    out_img = str(tmp_path / "risk.png")
    plot_risk_and_aggregator(
        csv_path=sample_csv_data["tara"],
        output_path=out_img
    )
    assert os.path.exists(out_img)
    assert os.path.getsize(out_img) > 1000


def test_plot_method_bar_comparison(tmp_path):
    out_img = str(tmp_path / "bar.png")
    summary = {
        "Baseline A: FedAvg": 52.4,
        "Baseline B: Trust+FedAvg": 71.8,
        "Baseline C: Robust Trimmed": 79.2,
        "Baseline D: TARA-FL Adaptive": 91.6
    }
    plot_method_bar_comparison(summary, output_path=out_img)
    assert os.path.exists(out_img)
    assert os.path.getsize(out_img) > 1000


def test_report_generator_markdown_and_html(sample_csv_data, tmp_path):
    reports_dir = str(tmp_path / "reports")
    gen = ReportGenerator(output_dir=reports_dir)

    fedavg_data = gen.load_experiment_csv(sample_csv_data["fedavg"])
    tara_data = gen.load_experiment_csv(sample_csv_data["tara"])

    assert len(fedavg_data) == 5
    assert len(tara_data) == 5

    baselines_data = {
        "FedAvg": fedavg_data,
        "TARA-FL Adaptive": tara_data
    }

    md_path = gen.generate_markdown_report("MNIST_Byzantine_Test", baselines_data)
    html_path = gen.generate_html_report("MNIST_Byzantine_Test", baselines_data)

    assert os.path.exists(md_path)
    assert os.path.exists(html_path)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "TARA-FL Experimental Evaluation Report" in md_text
        assert "Executive Summary & Attribution Matrix" in md_text

    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
        assert "<!DOCTYPE html>" in html_text
        assert "TARA-FL Benchmark Report" in html_text
        assert "accuracyChart" in html_text


def test_dashboard_handler_and_server(sample_csv_data):
    # Test RequestHandler logic directly
    DashboardRequestHandler.data_dir = sample_csv_data["dir"]
    handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
    handler.data_dir = sample_csv_data["dir"]

    runs = handler.list_available_runs()
    assert len(runs) >= 2

    metrics = handler.load_run_metrics(os.path.basename(sample_csv_data["tara"]))
    assert len(metrics["rounds"]) == 5
    assert "1" in metrics["trust"]
    assert "2" in metrics["trust"]

    html = handler.generate_dashboard_html()
    assert "TARA-FL System Telemetry Dashboard" in html

    # Test server spin-up and teardown in thread
    httpd = start_dashboard_server(data_dir=sample_csv_data["dir"], port=8899, run_in_thread=True)
    assert httpd is not None

    try:
        # Check HTTP response
        with urllib.request.urlopen("http://127.0.0.1:8899/api/status") as response:
            assert response.status == 200
            data = json.loads(response.read().decode())
            assert data["status"] == "running"
    finally:
        httpd.shutdown()
        httpd.server_close()
