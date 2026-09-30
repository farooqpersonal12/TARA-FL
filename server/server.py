import os
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.base_model import BaseFLModel
from model.model import MNISTModel, get_model
from detection.base_detector import BaseDetector
from detection.pid_detector import PIDDetector
from detection.robust_pid_detector import RobustPIDDetector
from detection.detector_factory import get_detector
from trustengine.trust_engine import TrustEngine
from risk.round_risk import RoundRisk
from risk.threat_classifier import ThreatType
from aggregation.adaptive_aggregator import AdaptiveAggregator


class Server:
    """
    Central Server Orchestrator for TARA-FL.

    Features:
      - Dynamic model architecture injection (MNIST, CIFAR-10, ResNet-18, MLP).
      - Pluggable anomaly detectors (Standard PID, Robust PID, Multi-Metric).
      - Continuous dynamic trust engine & quarantine management.
      - Multi-factor environmental round-risk assessment.
      - Adaptive aggregation routing & multi-model candidate evaluation.
      - Checkpoint persistence (weights + trust trajectories) and evaluation.
    """

    def __init__(
            self,
            model: Optional[Union[BaseFLModel, nn.Module, str]] = None,
            detector_type: str = "robust",
            enable_quarantine: bool = True,
            quarantine_threshold: float = 0.20,
            required_clean_rounds: int = 2,
            trim_ratio: float = 0.25,
            trust_floor: float = 0.05,
            probe_dataset: Optional[Dataset] = None,
            device: Optional[str] = None,
            **kwargs
    ):
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Global Model Initialization
        if model is None:
            self.global_model = MNISTModel()
        elif isinstance(model, str):
            self.global_model = get_model(model)
        elif isinstance(model, nn.Module):
            self.global_model = model
        else:
            raise TypeError(f"Invalid model specification: {type(model)}")

        self.global_model.to(self.device)

        # Filter kwargs for detector vs trust engine vs round risk
        trust_engine_param_names = {
            "history_weight", "current_weight", "persistence_weight",
            "decay_factor", "mad_scale", "minimum_scale", "normal_deviation"
        }
        round_risk_param_names = {
            "low_risk_threshold", "medium_risk_threshold",
            "weight_anom", "weight_trust", "weight_suspicious", "weight_worst",
            "suspicious_trust_threshold", "enable_performance_feedback"
        }
        trust_kwargs = {k: v for k, v in kwargs.items() if k in trust_engine_param_names}
        risk_kwargs = {k: v for k, v in kwargs.items() if k in round_risk_param_names}
        detector_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in trust_engine_param_names and k not in round_risk_param_names
        }

        # 2. Malicious / Anomaly Detector
        self.detector_type = detector_type
        if isinstance(detector_type, BaseDetector):
            self.detector = detector_type
        elif detector_type == "robust":
            self.detector = RobustPIDDetector(**detector_kwargs)
        elif detector_type in ["standard", "pid"]:
            self.detector = PIDDetector(**detector_kwargs)
        else:
            try:
                self.detector = get_detector(detector_type, **detector_kwargs)
            except Exception:
                self.detector = RobustPIDDetector(**detector_kwargs)

        # 3. Dynamic Client Trust Engine
        self.trust_engine = TrustEngine(
            enable_quarantine=enable_quarantine,
            quarantine_threshold=quarantine_threshold,
            required_clean_rounds=required_clean_rounds,
            **trust_kwargs
        )

        # 4. Round-Risk Assessment
        self.round_risk = RoundRisk(**risk_kwargs)

        # 5. Adaptive Aggregator
        self.adaptive_aggregator = AdaptiveAggregator(
            trim_ratio=trim_ratio,
            trust_floor=trust_floor,
            probe_dataset=probe_dataset
        )

    # ======================================================
    # ROUND RISK & THREAT HELPERS
    # ======================================================

    def calculate_round_risk(
            self,
            distances: Optional[Dict[int, float]] = None,
            trust_scores: Optional[Dict[int, float]] = None,
            current_accuracy: Optional[float] = None,
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> Tuple[float, str, int]:
        """
        Assess current round risk using the multi-factor RoundRisk engine.

        Returns:
            (risk_score, risk_level, suspicious_clients)
        """
        return self.round_risk.calculate_risk(
            distances=distances,
            trust_scores=trust_scores,
            current_accuracy=current_accuracy,
            client_updates=client_updates
        )

    def classify_threat(
            self,
            distances: Optional[Dict[int, float]] = None,
            trust_scores: Optional[Dict[int, float]] = None,
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> ThreatType:
        """Classify dominant threat signature for the current round."""
        return self.round_risk.classify_threat(
            distances=distances,
            trust_scores=trust_scores,
            client_updates=client_updates
        )

    def get_round_risk_details(self) -> Dict[str, Any]:
        """Get complete breakdown of the latest round risk calculation."""
        return self.round_risk.get_last_details()

    # ======================================================
    # ADAPTIVE AGGREGATION
    # ======================================================

    def aggregate(
            self,
            client_parameters: List[Dict[str, torch.Tensor]],
            client_sizes: List[int],
            trust_scores: Dict[int, float],
            risk_level: str = "LOW",
            client_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
            mode: str = "risk_routing",
            client_ids: Optional[List[int]] = None,
            quarantine_manager: Optional[Any] = None
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Execute adaptive aggregation with QuarantineManager enforcement.

        Workflow:
          Client Updates -> Trust Analysis -> QuarantineManager ->
          Determine aggregation-eligible clients -> Adaptive Aggregation -> Global Model
        """
        # 1. Resolve client IDs
        if client_ids is None:
            if client_updates is not None and len(client_updates) == len(client_parameters):
                client_ids = list(client_updates.keys())
            elif len(trust_scores) == len(client_parameters):
                client_ids = list(trust_scores.keys())
            else:
                client_ids = list(range(1, len(client_parameters) + 1))

        # 2. Determine QuarantineManager
        qm = quarantine_manager if quarantine_manager is not None else (
            self.trust_engine.quarantine_manager if hasattr(self, 'trust_engine') and self.trust_engine else None
        )

        # 3. Determine aggregation-eligible clients
        if qm is not None:
            eligible_indices = []
            eligible_client_ids = []
            quarantined_client_ids = []

            for i, cid in enumerate(client_ids):
                if qm.is_eligible_for_aggregation(cid):
                    eligible_indices.append(i)
                    eligible_client_ids.append(cid)
                else:
                    quarantined_client_ids.append(cid)

            if quarantined_client_ids:
                print(f"[QuarantineManager] Excluded {len(quarantined_client_ids)} quarantined/evicted client(s): {quarantined_client_ids}")
                print(f"[QuarantineManager] Active aggregation-eligible clients: {eligible_client_ids}")

            if len(eligible_indices) == 0:
                print("[QuarantineManager] [WARNING] All clients are quarantined! Retaining current global model.")
                return {k: v.clone().to(self.device) for k, v in self.global_model.state_dict().items()}, "QUARANTINE_ALL_RETAINED"

            agg_parameters = [client_parameters[i] for i in eligible_indices]
            agg_sizes = [client_sizes[i] for i in eligible_indices]
            agg_trust_scores = {cid: trust_scores.get(cid, 0.0) for cid in eligible_client_ids}
            agg_client_updates = {cid: client_updates[cid] for cid in eligible_client_ids} if client_updates is not None else None
            agg_client_ids = eligible_client_ids
        else:
            agg_parameters = client_parameters
            agg_sizes = client_sizes
            agg_trust_scores = trust_scores
            agg_client_updates = client_updates
            agg_client_ids = client_ids

        # 4. Dispatch to Adaptive Aggregator
        new_parameters, selected_aggregator = self.adaptive_aggregator.aggregate(
            client_parameters=agg_parameters,
            client_sizes=agg_sizes,
            trust_scores=agg_trust_scores,
            risk_level=risk_level,
            client_updates=agg_client_updates,
            model_template=self.global_model,
            mode=mode,
            client_ids=agg_client_ids
        )

        # Move aggregated weights to server device
        device_params = {k: v.to(self.device) for k, v in new_parameters.items()}

        print()
        print("Adaptive Aggregation")
        print("------------------------------")
        print(f"Round Risk: {risk_level}")
        print(f"Selected Aggregator: {selected_aggregator}")

        return device_params, selected_aggregator

    # ======================================================
    # GLOBAL MODEL EVALUATION
    # ======================================================

    def evaluate(
            self,
            test_dataset: Dataset,
            batch_size: int = 32,
            return_loss: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Evaluate global model on test dataset.
        
        Args:
            test_dataset: PyTorch Dataset
            batch_size: Evaluation mini-batch size
            return_loss: If True, returns (average_loss, accuracy). If False, returns accuracy.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.global_model.eval()
        loss_func = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.global_model(images)
                loss = loss_func(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / max(1, total)
        avg_loss = total_loss / max(1, total)

        if return_loss:
            return avg_loss, accuracy
        return accuracy

    # ======================================================
    # CHECKPOINT PERSISTENCE
    # ======================================================

    def save_checkpoint(self, filepath: str, round_number: int, extra_meta: Optional[Dict[str, Any]] = None):
        """Save server state (model weights + trust trajectories) to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "round_number": round_number,
            "model_state": {k: v.cpu() for k, v in self.global_model.state_dict().items()},
            "trust_history": self.trust_engine.history.to_dict(),
            "meta": extra_meta or {}
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load server state from disk and return saved round number."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint["model_state"])
        self.trust_engine.history.from_dict(checkpoint["trust_history"])
        return checkpoint.get("round_number", 0)