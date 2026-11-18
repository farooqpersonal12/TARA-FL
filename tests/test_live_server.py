"""
Integration tests: live dashboard backend (visualization/live_server.py).

Verifies the FastAPI backend serves the demo, reports backend/experiment
status honestly (404 + explicit error when no experiment data exists), and
streams the live experiment document over Server-Sent Events. Uses the real
ResultsExporter output as the data source - nothing is mocked at the API level.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from experiments.results_export import ResultsExporter, build_config_snapshot  # noqa: E402
from visualization.live_server import DashboardStore, create_app  # noqa: E402


def make_exporter(tmp_path):
    config = build_config_snapshot(
        args=type("A", (), {
            "num_clients": 2, "malicious_ratio": 0.5, "attack": "label_flip",
            "data_split": "iid", "rounds": 2, "local_epochs": 1, "seeds": [42],
            "detector_type": "standard", "flip_ratio": 0.75, "scale_factor": 10.0,
            "alpha": 0.5, "method": "tara", "output_dir": "experiments",
            "plots_dir": "plots", "reports_dir": "reports",
        })(),
        scenario_tag="server_test",
        methods_to_run=["tara"],
        num_malicious=1,
        malicious_ids=[2],
    )
    return ResultsExporter(dashboard_dir=str(tmp_path / "dashboard_data"), config=config)


def row(round_no):
    return {
        "method": "tara", "seed": 42, "round": round_no,
        "accuracy": 0.9, "accuracy_percent": 90.0, "loss": 0.4,
        "risk_score": 0.2, "risk_level": "LOW", "risk_details": None,
        "threat_type": "CLEAN", "suspicious_clients": 0, "malicious_weight": 0.0,
        "aggregator": "TRUST_AWARE_FEDAVG", "round_time_sec": 0.1,
    }


@pytest.fixture()
def client(tmp_path):
    store = DashboardStore(dashboard_dir=str(tmp_path / "dashboard_data"),
                           demo_dir=os.path.join(PROJECT_ROOT, "demo"))
    app = create_app(store)
    return TestClient(app), tmp_path


# ----------------------------------------------------------------------
# Status endpoint
# ----------------------------------------------------------------------

def test_status_online_without_data(client):
    tc, _ = client
    res = tc.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["backend"] == "online"
    assert body["data_available"] is False
    assert body["live"] is None  # no experiment -> reported honestly


def test_status_shows_running_experiment(client):
    tc, tmp_path = client
    exp = make_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=row(1), num_clients=2, malicious_ids=[2])
    body = tc.get("/api/status").json()
    assert body["data_available"] is True
    assert body["live"]["status"] == "running"
    assert body["live"]["rounds_completed"]["TARA"]["42"] == 1


# ----------------------------------------------------------------------
# Live document endpoint
# ----------------------------------------------------------------------

def test_live_404_when_no_experiment(client):
    tc, _ = client
    res = tc.get("/api/live")
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "no_live_experiment"


def test_live_returns_full_document(client):
    tc, tmp_path = client
    exp = make_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=row(1), num_clients=2, malicious_ids=[2])
    exp.finalize("completed")
    doc = tc.get("/api/live").json()
    assert doc["experiment_id"] == exp.experiment_id
    assert doc["status"] == "completed"
    assert doc["methods"]["TARA"]["runs"]["42"]["rounds"][0]["accuracy_percent"] == 90.0


# ----------------------------------------------------------------------
# History endpoints
# ----------------------------------------------------------------------

def test_experiments_index_empty_then_populated(client):
    tc, tmp_path = client
    body = tc.get("/api/experiments").json()
    assert body["count"] == 0 and body["experiments"] == []
    exp = make_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=row(1), num_clients=2, malicious_ids=[2])
    body = tc.get("/api/experiments").json()
    assert body["count"] == 1
    assert body["experiments"][0]["experiment_id"] == exp.experiment_id


def test_experiment_by_id_and_unknown_404(client):
    tc, tmp_path = client
    exp = make_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=row(1), num_clients=2, malicious_ids=[2])
    res = tc.get(f"/api/experiments/{exp.experiment_id}")
    assert res.status_code == 200
    assert res.json()["experiment_id"] == exp.experiment_id
    assert tc.get("/api/experiments/does-not-exist").status_code == 404


# ----------------------------------------------------------------------
# Dashboard + published static files
# ----------------------------------------------------------------------

def test_demo_index_served(client):
    tc, _ = client
    res = tc.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Real Results" in res.text


def test_published_results_404_is_explicit(client):
    tc, _ = client
    res = tc.get("/results/definitely_missing_<no-such-file>.json")
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "not_published"


def test_published_results_served_when_present(tmp_path):
    """Uses an isolated demo dir - must never touch the repository's real
    demo/results/ contents."""
    store = DashboardStore(dashboard_dir=str(tmp_path / "dashboard_data"),
                           demo_dir=str(tmp_path / "demo"))
    demo_results = tmp_path / "demo" / "results"
    demo_results.mkdir(parents=True)
    path = demo_results / "latest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"experiment_id": "static-test", "status": "completed"}, f)
    app = create_app(store)
    tc = TestClient(app)
    res = tc.get("/results/latest.json")
    assert res.status_code == 200
    assert res.json()["experiment_id"] == "static-test"
    assert path.exists()  # the file is served, not consumed


# ----------------------------------------------------------------------
# SSE stream - tested over a REAL socket via the stdlib fallback server
# (TestClient cannot consume infinite SSE generators; curl-verified for the
# FastAPI path which shares the same DashboardStore event logic).
# ----------------------------------------------------------------------

def _read_sse_frames(sock_file, count=1, timeout=20.0):
    """Read `count` SSE frames (terminated by a blank line) from the socket."""
    import time as _time
    frames, buf = [], b""
    deadline = _time.time() + timeout
    while len(frames) < count and _time.time() < deadline:
        line = sock_file.readline()
        if not line:
            break
        buf += line
        if buf.endswith(b"\n\n"):
            frames.append(buf.decode("utf-8"))
            buf = b""
    return frames


@pytest.fixture()
def sse_client(tmp_path):
    from visualization.live_server import make_stdlib_server
    store = DashboardStore(dashboard_dir=str(tmp_path / "dashboard_data"),
                           demo_dir=os.path.join(PROJECT_ROOT, "demo"))
    server = make_stdlib_server(store, "127.0.0.1", 0)
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield store, tmp_path, server
    server.shutdown()


def test_stream_sends_snapshot_with_current_experiment(sse_client):
    import socket
    store, tmp_path, server = sse_client
    exp = make_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=row(1), num_clients=2, malicious_ids=[2])

    host, port = server.server_address
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(b"GET /api/stream HTTP/1.1\r\nHost: localhost\r\n\r\n")
    sock_file = sock.makefile("rb")
    status = sock_file.readline().decode()
    assert "200" in status
    # skip headers until the blank line
    while True:
        line = sock_file.readline()
        if line in (b"\r\n", b"\n", b""):
            break
    frames = _read_sse_frames(sock_file, count=1, timeout=15)
    assert frames, "no SSE frame received"
    lines = frames[0].strip().split("\n")
    assert lines[0] == "event: snapshot"
    assert lines[1] == f"id: {exp.experiment_id}"
    data = json.loads(lines[2][len("data: "):])
    assert data["live"]["experiment_id"] == exp.experiment_id
    sock.close()


def test_stream_pushes_update_when_new_round_completes(sse_client):
    """A round completing while the dashboard is connected must arrive as an
    'update' event without reconnecting - the core live-mode requirement."""
    import socket
    import threading
    store, tmp_path, server = sse_client
    exp = make_exporter(tmp_path)
    exp.record_round(method="tara", seed=42, row=row(1), num_clients=2, malicious_ids=[2])

    host, port = server.server_address
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(b"GET /api/stream HTTP/1.1\r\nHost: localhost\r\n\r\n")
    sock_file = sock.makefile("rb")
    sock_file.readline()  # status line
    while sock_file.readline() not in (b"\r\n", b"\n", b""):
        pass  # headers

    # round 2 completes while the dashboard is connected
    def complete_round():
        exp.record_round(method="tara", seed=42, row=row(2), num_clients=2, malicious_ids=[2])
        exp.finalize("completed")
    threading.Thread(target=complete_round, daemon=True).start()

    frames = _read_sse_frames(sock_file, count=2, timeout=20)
    assert len(frames) == 2, f"expected snapshot+update, got {len(frames)} frame(s)"
    first_lines = frames[0].strip().split("\n")
    second_lines = frames[1].strip().split("\n")
    assert first_lines[0] == "event: snapshot"
    assert second_lines[0] == "event: update"
    data = json.loads(second_lines[2][len("data: "):])
    rounds = data["live"]["methods"]["TARA"]["runs"]["42"]["rounds"]
    assert len(rounds) == 2 and rounds[-1]["round"] == 2
    assert data["live"]["status"] == "completed"
    sock.close()
