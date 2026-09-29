import json
import os
from typing import Dict, List, Optional, Any
import torch


class TrustHistory:
    """
    Stores historical trust, anomaly, and quarantine records for each client
    across federated learning rounds.
    
    Supports JSON serialization and disk checkpointing.
    """

    def __init__(self):
        self.trust_history: Dict[int, List[float]] = {}
        self.anomaly_history: Dict[int, List[float]] = {}
        self.probe_history: Dict[int, List[float]] = {}
        self.quarantine_history: Dict[int, List[bool]] = {}

    # ---------------------------------------------------------
    # Trust history
    # ---------------------------------------------------------

    def update_trust(self, client_id: int, trust_score: float):
        if client_id not in self.trust_history:
            self.trust_history[client_id] = []
        self.trust_history[client_id].append(float(trust_score))

    def get_latest_trust(self, client_id: int) -> Optional[float]:
        history = self.trust_history.get(client_id, [])
        if not history:
            return None
        return history[-1]

    def get_trust_history(self, client_id: int) -> List[float]:
        return list(self.trust_history.get(client_id, []))

    # ---------------------------------------------------------
    # Anomaly history
    # ---------------------------------------------------------

    def update_anomaly(self, client_id: int, anomaly_score: float):
        if client_id not in self.anomaly_history:
            self.anomaly_history[client_id] = []
        self.anomaly_history[client_id].append(float(anomaly_score))

    def get_anomaly_history(self, client_id: int) -> List[float]:
        return list(self.anomaly_history.get(client_id, []))

    # ---------------------------------------------------------
    # Probe history
    # ---------------------------------------------------------

    def update_probe(self, client_id: int, probe_score: float):
        if client_id not in self.probe_history:
            self.probe_history[client_id] = []
        self.probe_history[client_id].append(float(probe_score))

    def get_probe_history(self, client_id: int) -> List[float]:
        return list(self.probe_history.get(client_id, []))

    # ---------------------------------------------------------
    # Quarantine tracking
    # ---------------------------------------------------------

    def record_quarantine_status(self, client_id: int, is_quarantined: bool):
        if client_id not in self.quarantine_history:
            self.quarantine_history[client_id] = []
        self.quarantine_history[client_id].append(bool(is_quarantined))

    def get_quarantine_count(self, client_id: int) -> int:
        return sum(1 for q in self.quarantine_history.get(client_id, []) if q)

    # ---------------------------------------------------------
    # Combined update
    # ---------------------------------------------------------

    def update(
            self,
            client_id: int,
            trust_score: float,
            anomaly_score: float,
            probe_score: Optional[float] = None
    ):
        self.update_trust(client_id, trust_score)
        self.update_anomaly(client_id, anomaly_score)
        if probe_score is not None:
            self.update_probe(client_id, probe_score)

    # ---------------------------------------------------------
    # Serialization / Persistence
    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trust_history": {str(k): v for k, v in self.trust_history.items()},
            "anomaly_history": {str(k): v for k, v in self.anomaly_history.items()},
            "probe_history": {str(k): v for k, v in self.probe_history.items()},
            "quarantine_history": {str(k): v for k, v in self.quarantine_history.items()},
        }

    def from_dict(self, data: Dict[str, Any]):
        self.trust_history = {int(k): v for k, v in data.get("trust_history", {}).items()}
        self.anomaly_history = {int(k): v for k, v in data.get("anomaly_history", {}).items()}
        self.probe_history = {int(k): v for k, v in data.get("probe_history", {}).items()}
        self.quarantine_history = {int(k): v for k, v in data.get("quarantine_history", {}).items()}

    def save_to_json(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_json(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.from_dict(data)

    def clear(self):
        self.trust_history.clear()
        self.anomaly_history.clear()
        self.probe_history.clear()
        self.quarantine_history.clear()