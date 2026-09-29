from typing import Dict, Type, List
from detection.base_detector import BaseDetector
from detection.pid_detector import PIDDetector
from detection.robust_pid_detector import RobustPIDDetector
from detection.multimetric_detector import MultiMetricPIDDetector


class DetectorFactory:
    """Factory to instantiate anomaly detectors by string key."""

    _registry: Dict[str, Type[BaseDetector]] = {
        "standard": PIDDetector,
        "pid": PIDDetector,
        "robust": RobustPIDDetector,
        "robust_pid": RobustPIDDetector,
        "multimetric": MultiMetricPIDDetector,
        "multi_metric": MultiMetricPIDDetector,
    }

    @classmethod
    def register(cls, name: str, detector_class: Type[BaseDetector]):
        cls._registry[name.lower().strip()] = detector_class

    @classmethod
    def list_detectors(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDetector:
        key = name.lower().strip()
        if key not in cls._registry:
            available = ", ".join(cls.list_detectors())
            raise ValueError(f"Unknown detector '{name}'. Available: [{available}]")
        return cls._registry[key](**kwargs)


def get_detector(name: str = "robust", **kwargs) -> BaseDetector:
    """Convenience helper to retrieve an anomaly detector."""
    return DetectorFactory.create(name, **kwargs)
