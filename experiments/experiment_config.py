from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ExperimentConfig:
    """Configuration specification for running TARA-FL and baseline FL experiments."""
    experiment_name: str = "tara_benchmark"
    dataset_name: str = "mnist"
    model_name: str = "mnist"
    num_clients: int = 10
    num_malicious: int = 2
    attack_type: str = "label_flip"
    flip_ratio: float = 0.5
    scale_factor: float = 10.0
    is_noniid: bool = False
    alpha: float = 0.5
    num_rounds: int = 10
    local_epochs: int = 1
    lr: float = 0.01
    batch_size: int = 32
    detector_type: str = "robust"
    enable_quarantine: bool = True
    seeds: List[int] = field(default_factory=lambda: [42])
    output_dir: str = "experiments/results"
