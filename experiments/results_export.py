"""
TARA-FL Experiment Results Exporter.

Bridges the real Python federated training pipeline (experiments/run_unified.py)
to the interactive browser dashboard (demo/index.html).

Design goals
------------
1. SINGLE SOURCE OF TRUTH: every number written here comes directly from the
   running experiment (accuracy/loss from server.evaluate, trust scores from
   TrustEngine.calculate_trust, risk from RoundRisk, aggregator from the
   AdaptiveAggregator router). Nothing is estimated or synthesized.

2. PROGRESSIVE PUBLISHING: the exporter is called after EVERY completed
   communication round and atomically rewrites:
       dashboard_data/live.json              -> the current / latest run
       dashboard_data/history/<exp_id>.json  -> persistent per-experiment copy
       dashboard_data/index.json             -> experiment history index
   Because each write is atomic (tmp file + os.replace) and happens after the
   round is complete, a crash or Ctrl+C leaves all completed rounds intact on
   disk (status "running"/"stopped"), and the FastAPI live server streams each
   update to connected dashboards via Server-Sent Events.

3. TRACEABILITY: every round payload carries the experiment id, method, seed
   and round number, so any dashboard metric can be traced back to the exact
   Python experiment and round that produced it.

JSON schema (v1.1) - see docs in README.md section "Real experiment data format".
"""

import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.1"

# Zone strings produced by QuarantineManager / TrustEngine.get_trust_zone
KNOWN_ZONES = {"HIGH", "MEDIUM", "LOW", "QUARANTINE", "EVICTED", "NORMAL"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_zone(zone: Any) -> Optional[str]:
    """Normalize trust-zone values: Python 3.11+ str(Enum) yields
    'TrustZone.HIGH'; older paths and get_trust_zone yield plain 'HIGH'.
    Store the bare zone name either way."""
    if zone is None:
        return None
    text = str(zone)
    if text.startswith("TrustZone."):
        text = text[len("TrustZone."):]
    return text


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy / torch scalars and NaN/Inf into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int,)):
        return int(value)
    try:
        if isinstance(value, float):
            return value if value == value and abs(value) != float("inf") else None
    except Exception:
        return None
    if hasattr(value, "item"):  # numpy / torch scalar
        try:
            return _json_safe(value.item())
        except Exception:
            return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically: dump to <path>.tmp then os.replace onto <path>.

    Atomic replace guarantees a concurrent reader (the dashboard backend)
    never observes a partially written file.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return slug or "experiment"


class ResultsExporter:
    """Progressive JSON exporter for a TARA-FL benchmark invocation.

    One exporter instance covers one full run of experiments/run_unified.py
    (which may execute several methods and seeds). Each (method, seed) pair is
    recorded as a separate "run" inside the document.
    """

    def __init__(
            self,
            dashboard_dir: str = "dashboard_data",
            config: Optional[Dict[str, Any]] = None,
            source_info: Optional[Dict[str, Any]] = None,
            experiment_id: Optional[str] = None,
    ):
        self.dashboard_dir = dashboard_dir
        self.config = _json_safe(config or {})
        self.source_info = _json_safe(source_info or {})

        scenario = _slugify(str(self.config.get("scenario_tag", "experiment")))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_id = experiment_id or f"exp-{stamp}-{scenario}"

        self.created_at = _utc_now_iso()
        self.updated_at = self.created_at
        self.finalized_at: Optional[str] = None
        self.status = "running"

        # Per (method, seed) last-seen quarantine state, for transition detection
        self._prev_zone_state: Dict[str, Dict[int, Dict[str, Any]]] = {}

        self.doc: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finalized_at": None,
            "config": self.config,
            "environment": self._collect_environment(),
            "source": self.source_info,
            "methods": {},
            "notes": (
                "All metrics in this file were produced by the actual Python "
                "TARA-FL pipeline (server.evaluate, TrustEngine, RoundRisk, "
                "AdaptiveAggregator). No values are simulated or estimated."
            ),
        }

        # Publish the initial "running" snapshot so the dashboard detects the
        # experiment the moment training starts (even before round 1 finishes).
        self._flush()

    # ------------------------------------------------------------------
    # Environment / provenance
    # ------------------------------------------------------------------

    def _collect_environment(self) -> Dict[str, Any]:
        env = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "dataset": str(self.config.get("dataset", "MNIST")),
        }
        try:
            import torch
            env["torch_version"] = torch.__version__
            env["torch_threads"] = torch.get_num_threads()
            env["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            env["torch_version"] = "unavailable"
        return env

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def live_path(self) -> str:
        return os.path.join(self.dashboard_dir, "live.json")

    @property
    def history_path(self) -> str:
        return os.path.join(self.dashboard_dir, "history", f"{self.experiment_id}.json")

    @property
    def index_path(self) -> str:
        return os.path.join(self.dashboard_dir, "index.json")

    # ------------------------------------------------------------------
    # Round recording (called once per completed communication round)
    # ------------------------------------------------------------------

    def record_round(
            self,
            method: str,
            seed: int,
            row: Dict[str, Any],
            num_clients: int,
            malicious_ids: List[int],
            trust_details: Optional[Dict[int, Dict[str, Any]]] = None,
            distances: Optional[Dict[int, float]] = None,
            pid_scores: Optional[Dict[int, float]] = None,
            quarantine_manager: Any = None,
    ) -> None:
        """Record one completed communication round of a (method, seed) run.

        `row` is the exact per-round dictionary already produced by
        run_experiment (accuracy/loss/risk/aggregator/...); trust_details is the
        full per-client output of TrustEngine.calculate_trust when the method
        computed trust (fedavg does not -> None).
        """
        method_key = str(method).upper()
        run_key = str(seed)

        method_doc = self.doc["methods"].setdefault(method_key, {
            "runs": {},
            "summary": None,
        })
        run_doc = method_doc["runs"].setdefault(run_key, {"rounds": [], "summary": None})

        trust_available = bool(trust_details)
        clients_payload: Dict[str, Any] = {}
        zone_state: Dict[int, Dict[str, Any]] = {}
        for cid in range(1, num_clients + 1):
            detail = trust_details.get(cid) if trust_details else None
            if detail is None:
                clients_payload[str(cid)] = {
                    "trust": None,
                    "pid_score": None,
                    "relative_anomaly": None,
                    "zone": None,
                    "is_quarantined": None,
                    "eligible_for_aggregation": None,
                }
                continue
            quarantined = bool(detail.get("is_quarantined", False))
            if quarantine_manager is not None:
                try:
                    quarantined = bool(quarantine_manager.is_quarantined(cid))
                    eligible = bool(quarantine_manager.is_eligible_for_aggregation(cid))
                except Exception:
                    eligible = not quarantined
            else:
                eligible = (None if not trust_available else not quarantined)
            zone = _normalize_zone(detail.get("zone"))
            clients_payload[str(cid)] = {
                "trust": detail.get("trust"),
                "pid_score": detail.get("pid_score"),
                "relative_anomaly": detail.get("relative_anomaly"),
                "current_trust": detail.get("current_trust"),
                "historical_trust": detail.get("historical_trust"),
                "persistence": detail.get("persistence"),
                "zone": zone,
                "is_quarantined": quarantined,
                "eligible_for_aggregation": eligible,
            }
            zone_state[cid] = {"zone": zone, "quarantined": quarantined}

        # Quarantine / zone transitions versus the previous round of this run
        state_key = f"{method_key}:{run_key}"
        transitions: List[Dict[str, Any]] = []
        prev_state = self._prev_zone_state.get(state_key)
        if prev_state:
            for cid, cur in zone_state.items():
                prev = prev_state.get(cid)
                if not prev:
                    continue
                if cur["zone"] != prev["zone"] or cur["quarantined"] != prev["quarantined"]:
                    transitions.append({
                        "client_id": cid,
                        "from_zone": prev["zone"],
                        "to_zone": cur["zone"],
                        "quarantined": cur["quarantined"],
                        "previously_quarantined": prev["quarantined"],
                    })
        self._prev_zone_state[state_key] = zone_state

        # Ground-truth malicious clients currently quarantined (defense outcome)
        malicious_quarantined = None
        if trust_available:
            malicious_quarantined = sum(
                1 for cid in malicious_ids
                if cid in zone_state and zone_state[cid]["quarantined"]
            )

        round_payload = {
            "round": int(row.get("round", len(run_doc["rounds"]) + 1)),
            "seed": int(seed),
            "accuracy": row.get("accuracy"),
            "accuracy_percent": row.get("accuracy_percent"),
            "loss": row.get("loss"),
            "risk_score": row.get("risk_score"),
            "risk_level": row.get("risk_level"),
            "risk_details": _json_safe(row.get("risk_details")),
            "threat_type": row.get("threat_type"),
            "aggregator": row.get("aggregator"),
            "suspicious_clients": row.get("suspicious_clients"),
            "malicious_weight": row.get("malicious_weight"),
            "round_time_sec": row.get("round_time_sec"),
            "trust_available": trust_available,
            "malicious_quarantined": malicious_quarantined,
            "quarantine_transitions": transitions,
            "clients": clients_payload,
            "detection": {
                "distances": _json_safe(distances) if distances else None,
                "pid_scores": _json_safe(pid_scores) if pid_scores else None,
            },
        }

        # Replace-in-place if this round was already recorded (re-run safety)
        existing = [r for r in run_doc["rounds"] if r["round"] == round_payload["round"]]
        if existing:
            run_doc["rounds"][run_doc["rounds"].index(existing[0])] = round_payload
        else:
            run_doc["rounds"].append(round_payload)
        run_doc["rounds"].sort(key=lambda r: r["round"])

        self.updated_at = _utc_now_iso()
        self.doc["updated_at"] = self.updated_at
        self._flush()

    # ------------------------------------------------------------------
    # Run / method / experiment finalization
    # ------------------------------------------------------------------

    def finish_run(self, method: str, seed: int, summary: Dict[str, Any]) -> None:
        """Attach the per-seed summary (computed from real rounds by the runner)."""
        method_doc = self.doc["methods"].setdefault(str(method).upper(), {"runs": {}, "summary": None})
        run_doc = method_doc["runs"].setdefault(str(seed), {"rounds": [], "summary": None})
        run_doc["summary"] = _json_safe(summary)
        self.updated_at = _utc_now_iso()
        self.doc["updated_at"] = self.updated_at
        self._flush()

    def finish_method(self, method: str, summary: Dict[str, Any]) -> None:
        """Attach the across-seeds method summary (real values from the runner)."""
        method_doc = self.doc["methods"].setdefault(str(method).upper(), {"runs": {}, "summary": None})
        method_doc["summary"] = _json_safe(summary)
        self.updated_at = _utc_now_iso()
        self.doc["updated_at"] = self.updated_at
        self._flush()

    def finalize(self, status: str = "completed") -> None:
        """Mark the experiment finished ("completed" | "stopped" | "failed").

        Completed rounds are already on disk from the per-round flushes, so a
        finalize call only updates status/timestamps - an experiment killed
        without finalize still preserves every completed round.
        """
        self.status = status
        self.finalized_at = _utc_now_iso()
        self.doc["status"] = status
        self.doc["finalized_at"] = self.finalized_at
        self._flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _index_entry(self) -> Dict[str, Any]:
        method_summaries = {}
        for m_key, m_doc in self.doc["methods"].items():
            summary = m_doc.get("summary")
            if summary:
                method_summaries[m_key] = {
                    "mean_final_accuracy": summary.get("mean_final_accuracy"),
                    "mean_final_loss": summary.get("mean_final_loss"),
                }
        cfg = self.config
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finalized_at": self.finalized_at,
            "config_summary": {
                "scenario_tag": cfg.get("scenario_tag"),
                "methods": cfg.get("methods_to_run"),
                "seed": cfg.get("seeds"),
                "num_clients": cfg.get("num_clients"),
                "num_rounds": cfg.get("rounds"),
                "attack": cfg.get("attack"),
                "malicious_ratio": cfg.get("malicious_ratio"),
                "data_split": cfg.get("data_split"),
            },
            "method_summaries": method_summaries,
        }

    def _flush(self) -> None:
        """Atomically publish live.json + history/<id>.json + index.json."""
        _atomic_write_json(self.live_path, self.doc)
        _atomic_write_json(self.history_path, self.doc)
        self._update_index()

    def _update_index(self) -> None:
        """Rebuild the history index from the actual history files on disk."""
        history_dir = os.path.join(self.dashboard_dir, "history")
        entries: List[Dict[str, Any]] = []
        if os.path.isdir(history_dir):
            for name in sorted(os.listdir(history_dir)):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(history_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                except Exception:
                    continue  # unreadable / partially written files are skipped
                if doc.get("experiment_id") == self.experiment_id:
                    entries.append(self._index_entry())
                else:
                    entries.append({
                        "experiment_id": doc.get("experiment_id"),
                        "status": doc.get("status"),
                        "created_at": doc.get("created_at"),
                        "updated_at": doc.get("updated_at"),
                        "finalized_at": doc.get("finalized_at"),
                        "config_summary": (doc.get("config") or {}).get("scenario_tag")
                            and {
                                "scenario_tag": (doc.get("config") or {}).get("scenario_tag"),
                                "num_clients": (doc.get("config") or {}).get("num_clients"),
                                "num_rounds": (doc.get("config") or {}).get("rounds"),
                                "attack": (doc.get("config") or {}).get("attack"),
                                "malicious_ratio": (doc.get("config") or {}).get("malicious_ratio"),
                                "data_split": (doc.get("config") or {}).get("data_split"),
                            } or {},
                        "method_summaries": {
                            m: {
                                "mean_final_accuracy": (doc["methods"][m].get("summary") or {}).get("mean_final_accuracy"),
                                "mean_final_loss": (doc["methods"][m].get("summary") or {}).get("mean_final_loss"),
                            }
                            for m in (doc.get("methods") or {})
                            if doc["methods"][m].get("summary")
                        },
                    })
        entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        index_doc = {
            "updated_at": _utc_now_iso(),
            "count": len(entries),
            "experiments": entries,
        }
        _atomic_write_json(self.index_path, index_doc)


def build_config_snapshot(args: Any, scenario_tag: str, methods_to_run: List[str],
                          num_malicious: int, malicious_ids: List[int]) -> Dict[str, Any]:
    """Capture the complete experiment configuration for reproducibility."""
    return {
        "scenario_tag": scenario_tag,
        "methods_to_run": methods_to_run,
        "method": getattr(args, "method", None),
        "seed": getattr(args, "seeds", None),
        "seeds": getattr(args, "seeds", None),
        "dataset": "MNIST",
        "num_clients": getattr(args, "num_clients", None),
        "num_malicious": num_malicious,
        "malicious_client_ids": malicious_ids,
        "malicious_ratio": getattr(args, "malicious_ratio", None),
        "num_rounds": getattr(args, "rounds", None),
        "rounds": getattr(args, "rounds", None),
        "local_epochs": getattr(args, "local_epochs", None),
        "attack": getattr(args, "attack", None),
        "flip_ratio": getattr(args, "flip_ratio", None),
        "scale_factor": getattr(args, "scale_factor", None),
        "data_split": getattr(args, "data_split", None),
        "alpha": getattr(args, "alpha", None),
        "detector_type": getattr(args, "detector_type", None),
        "output_dir": getattr(args, "output_dir", None),
        "plots_dir": getattr(args, "plots_dir", None),
        "reports_dir": getattr(args, "reports_dir", None),
    }
