"""
Integration tests: real experiment results export (experiments/results_export.py).

Verifies that the exporter produces a complete, traceable, progressively
published JSON record of an actual experiment run, that completed rounds are
preserved on unexpected termination, and that nothing is fabricated.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.results_export import ResultsExporter, SCHEMA_VERSION, build_config_snapshot  # noqa: E402


REQUIRED_TOP_LEVEL = [
    "schema_version", "experiment_id", "status", "created_at", "updated_at",
    "finalized_at", "config", "environment", "source", "methods", "notes",
]

REQUIRED_ROUND_FIELDS = [
    "round", "seed", "accuracy", "accuracy_percent", "loss", "risk_score",
    "risk_level", "threat_type", "aggregator", "suspicious_clients",
    "malicious_weight", "round_time_sec", "trust_available",
    "malicious_quarantined", "quarantine_transitions", "clients", "detection",
]


def make_row(round_no: int, trust: float = 0.9) -> dict:
    return {
        "method": "tara", "seed": 42, "round": round_no,
        "accuracy": trust, "accuracy_percent": trust * 100, "loss": 2.5 - trust,
        "risk_score": 0.4, "risk_level": "MEDIUM", "risk_details": {"risk_score": 0.4},
        "threat_type": "MAGNITUDE_SCALING", "suspicious_clients": 2,
        "malicious_weight": 0.03, "aggregator": "TRUST_WEIGHTED_ROBUST",
        "round_time_sec": 1.5,
        "trust_client_1": trust, "trust_client_2": trust,
    }


def make_trust_details(zone: str = "HIGH") -> dict:
    return {
        1: {"client_id": 1, "pid_score": 0.01, "relative_anomaly": 1.0,
            "current_trust": 1.0, "historical_trust": 1.0, "persistence": 0.0,
            "trust": 0.95, "zone": zone, "is_quarantined": False},
        2: {"client_id": 2, "pid_score": 0.40, "relative_anomaly": 3.5,
            "current_trust": 0.1, "historical_trust": 0.9, "persistence": 0.8,
            "trust": 0.05, "zone": "QUARANTINE", "is_quarantined": True},
    }


class FakeQuarantineManager:
    """Minimal stand-in mirroring QuarantineManager's public read API."""

    def __init__(self, quarantined):
        self.quarantined = set(quarantined)

    def is_quarantined(self, cid):
        return cid in self.quarantined

    def is_eligible_for_aggregation(self, cid):
        return cid not in self.quarantined


def new_exporter(tmp_path):
    config = build_config_snapshot(
        args=type("A", (), {
            "num_clients": 2, "malicious_ratio": 0.5, "attack": "gradient_scale",
            "data_split": "iid", "rounds": 3, "local_epochs": 1, "seeds": [42],
            "detector_type": "standard", "flip_ratio": 0.75, "scale_factor": 10.0,
            "alpha": 0.5, "method": "tara", "output_dir": "experiments",
            "plots_dir": "plots", "reports_dir": "reports",
        })(),
        scenario_tag="test_scenario",
        methods_to_run=["tara"],
        num_malicious=1,
        malicious_ids=[2],
    )
    return ResultsExporter(
        dashboard_dir=str(tmp_path / "dashboard_data"),
        config=config,
        source_info={"script": "experiments/run_unified.py", "argv": ["--rounds", "3"]},
    )


def read_live(tmp_path):
    with open(tmp_path / "dashboard_data" / "live.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Schema completeness
# ----------------------------------------------------------------------

def test_schema_complete_after_one_round(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    exp.finalize("completed")

    doc = read_live(tmp_path)
    for key in REQUIRED_TOP_LEVEL:
        assert key in doc, f"missing top-level field: {key}"
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["status"] == "completed"
    assert doc["finalized_at"] is not None

    rnd = doc["methods"]["TARA"]["runs"]["42"]["rounds"][0]
    for key in REQUIRED_ROUND_FIELDS:
        assert key in rnd, f"missing round field: {key}"
    # real trust detail preserved per client
    assert rnd["clients"]["2"]["zone"] == "QUARANTINE"
    assert rnd["clients"]["2"]["is_quarantined"] is True
    assert rnd["malicious_quarantined"] == 1
    assert rnd["detection"] is not None or rnd["detection"] == {"distances": None, "pid_scores": None}


def test_config_snapshot_is_reproducible(tmp_path):
    exp = new_exporter(tmp_path)
    cfg = exp.doc["config"]
    for key in ["num_clients", "num_rounds", "attack", "malicious_ratio",
                "data_split", "seeds", "malicious_client_ids", "dataset",
                "detector_type", "local_epochs"]:
        assert key in cfg, f"config missing {key}"
    assert cfg["dataset"] == "MNIST"
    assert cfg["malicious_client_ids"] == [2]


# ----------------------------------------------------------------------
# Progressive publishing + traceability
# ----------------------------------------------------------------------

def test_rounds_published_progressively(tmp_path):
    exp = new_exporter(tmp_path)
    for r in range(1, 4):
        exp.record_round(method="tara", seed=42, row=make_row(r), num_clients=2,
                         malicious_ids=[2], trust_details=make_trust_details())
        doc = read_live(tmp_path)
        # after each round, the on-disk file already contains it (SSE feed basis)
        assert len(doc["methods"]["TARA"]["runs"]["42"]["rounds"]) == r
        assert doc["status"] == "running"


def test_every_round_traceable(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    doc = read_live(tmp_path)
    rnd = doc["methods"]["TARA"]["runs"]["42"]["rounds"][0]
    assert rnd["round"] == 1
    assert rnd["seed"] == 42
    assert doc["experiment_id"].startswith("exp-")
    assert doc["source"]["script"] == "experiments/run_unified.py"


def test_history_file_written_alongside_live(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    hist = tmp_path / "dashboard_data" / "history" / f"{exp.experiment_id}.json"
    assert hist.is_file()
    with open(hist, "r", encoding="utf-8") as f:
        assert json.load(f)["experiment_id"] == exp.experiment_id


def test_index_lists_experiment(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    with open(tmp_path / "dashboard_data" / "index.json", "r", encoding="utf-8") as f:
        index = json.load(f)
    assert index["count"] == 1
    assert index["experiments"][0]["experiment_id"] == exp.experiment_id


def test_no_tmp_files_left_behind(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    leftovers = [p for p in os.listdir(tmp_path / "dashboard_data") if p.endswith(".tmp")]
    assert leftovers == []


# ----------------------------------------------------------------------
# Crash safety (requirement: preserve completed rounds on unexpected stop)
# ----------------------------------------------------------------------

def test_completed_rounds_survive_without_finalize(tmp_path):
    exp = new_exporter(tmp_path)
    for r in range(1, 3):
        exp.record_round(method="tara", seed=42, row=make_row(r), num_clients=2,
                         malicious_ids=[2], trust_details=make_trust_details())
    # simulate a crash: process dies without finalize() - read the file as-is
    doc = read_live(tmp_path)
    assert doc["status"] == "running"
    assert len(doc["methods"]["TARA"]["runs"]["42"]["rounds"]) == 2
    assert doc["methods"]["TARA"]["runs"]["42"]["rounds"][-1]["round"] == 2


def test_finalize_records_stop_status(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    exp.finalize("stopped")
    doc = read_live(tmp_path)
    assert doc["status"] == "stopped"
    assert len(doc["methods"]["TARA"]["runs"]["42"]["rounds"]) == 1


# ----------------------------------------------------------------------
# Honesty: no fabricated data
# ----------------------------------------------------------------------

def test_fedavg_records_null_trust_not_fabricated(tmp_path):
    exp = new_exporter(tmp_path)
    row = make_row(1)
    row["aggregator"] = "STANDARD_FEDAVG"
    row["risk_level"] = "N/A"
    exp.record_round(method="fedavg", seed=42, row=row, num_clients=2,
                     malicious_ids=[2], trust_details=None, distances=None,
                     pid_scores=None, quarantine_manager=None)
    doc = read_live(tmp_path)
    rnd = doc["methods"]["FEDAVG"]["runs"]["42"]["rounds"][0]
    assert rnd["trust_available"] is False
    assert rnd["clients"]["1"]["trust"] is None
    assert rnd["clients"]["1"]["zone"] is None
    assert rnd["malicious_quarantined"] is None  # no quarantine engine for fedavg


def test_quarantine_transitions_detected_between_rounds(tmp_path):
    exp = new_exporter(tmp_path)
    qm = FakeQuarantineManager(quarantined=[])
    # round 1: client 2 healthy HIGH
    details = make_trust_details(zone="HIGH")
    details[2] = dict(details[2], zone="HIGH", is_quarantined=False)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=details, quarantine_manager=qm)
    # round 2: client 2 drops into quarantine (client 1 stays HIGH)
    qm.quarantined.add(2)
    details2 = make_trust_details(zone="QUARANTINE")
    details2[1] = dict(details2[1], zone="HIGH", is_quarantined=False)
    exp.record_round(method="tara", seed=42, row=make_row(2), num_clients=2,
                     malicious_ids=[2], trust_details=details2, quarantine_manager=qm)
    doc = read_live(tmp_path)
    rounds = doc["methods"]["TARA"]["runs"]["42"]["rounds"]
    transitions = rounds[1]["quarantine_transitions"]
    assert len(transitions) == 1
    t = transitions[0]
    assert t["client_id"] == 2
    assert t["from_zone"] == "HIGH" and t["to_zone"] == "QUARANTINE"
    assert t["quarantined"] is True


def test_no_transitions_recorded_for_first_round(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    doc = read_live(tmp_path)
    assert doc["methods"]["TARA"]["runs"]["42"]["rounds"][0]["quarantine_transitions"] == []


def test_multiple_methods_and_seeds_isolated(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    exp.record_round(method="fedavg", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=None)
    exp.record_round(method="tara", seed=7, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    doc = read_live(tmp_path)
    assert set(doc["methods"].keys()) == {"TARA", "FEDAVG"}
    assert set(doc["methods"]["TARA"]["runs"].keys()) == {"42", "7"}


def test_summary_attaches_to_method(tmp_path):
    exp = new_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=make_row(1), num_clients=2,
                     malicious_ids=[2], trust_details=make_trust_details())
    exp.finish_run("tara", 42, {"final_accuracy": 90.0})
    exp.finish_method("TARA", {"mean_final_accuracy": 90.0, "mean_final_loss": 0.4})
    doc = read_live(tmp_path)
    m = doc["methods"]["TARA"]
    assert m["runs"]["42"]["summary"]["final_accuracy"] == 90.0
    assert m["summary"]["mean_final_accuracy"] == 90.0
