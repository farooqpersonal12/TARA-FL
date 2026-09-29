"""
TARA-FL Detection Package

Provides BaseDetector, standard PIDDetector, RobustPIDDetector (median reference),
MultiMetricPIDDetector (Euclidean + Cosine + Layerwise), and DetectorFactory.
"""

from detection.base_detector import BaseDetector
from detection.pid_detector import PIDDetector
from detection.robust_pid_detector import RobustPIDDetector
from detection.multimetric_detector import MultiMetricPIDDetector
from detection.detector_factory import DetectorFactory, get_detector

__all__ = [
    "BaseDetector",
    "PIDDetector",
    "RobustPIDDetector",
    "MultiMetricPIDDetector",
    "DetectorFactory",
    "get_detector"
]
