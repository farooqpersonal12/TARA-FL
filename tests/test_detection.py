import pytest
import torch
from detection.pid_detector import PIDDetector
from detection.robust_pid_detector import RobustPIDDetector
from detection.multimetric_detector import MultiMetricPIDDetector
from detection.detector_factory import DetectorFactory, get_detector


@pytest.fixture
def sample_client_updates():
    """Create sample client updates with 3 honest and 1 scaled adversary."""
    return {
        1: {"w": torch.tensor([1.0, 1.1, 0.9]), "b": torch.tensor([0.1])},
        2: {"w": torch.tensor([1.05, 0.95, 1.0]), "b": torch.tensor([0.12])},
        3: {"w": torch.tensor([0.98, 1.02, 1.01]), "b": torch.tensor([0.09])},
        4: {"w": torch.tensor([10.0, 15.0, 20.0]), "b": torch.tensor([5.0])},  # Adversary
    }


class TestDetectionSuite:
    """Test suite for anomaly and PID detection engines."""

    def test_standard_pid_detector(self, sample_client_updates):
        detector = PIDDetector(kp=1.0, ki=0.08, kd=5.0)
        distances, scores = detector.calculate_scores(sample_client_updates)

        assert len(distances) == 4
        assert len(scores) == 4
        # Adversary (Client 4) should have significantly higher score and distance
        assert distances[4] > distances[1]
        assert scores[4] > scores[1]

    def test_robust_pid_detector_median_resilience(self, sample_client_updates):
        detector = RobustPIDDetector(kp=1.0, ki=0.08, kd=5.0)
        distances, scores = detector.calculate_scores(sample_client_updates)

        assert distances[4] > distances[1]
        assert scores[4] > scores[1]

    def test_sliding_window_and_anti_windup(self, sample_client_updates):
        detector = PIDDetector(window_size=2, integral_cap=5.0)
        for _ in range(5):
            detector.calculate_scores(sample_client_updates)

        history = detector.get_client_history(1)
        assert len(history) == 5

    def test_multimetric_pid_detector(self, sample_client_updates):
        detector = MultiMetricPIDDetector(
            weight_euclidean=0.4,
            weight_cosine=0.4,
            weight_layerwise=0.2
        )
        distances, scores = detector.calculate_scores(sample_client_updates)

        assert len(distances) == 4
        assert len(scores) == 4
        assert distances[4] > distances[1]

    def test_detector_factory_and_get_detector(self):
        detectors = DetectorFactory.list_detectors()
        assert "standard" in detectors
        assert "robust" in detectors
        assert "multimetric" in detectors

        d_robust = get_detector("robust")
        assert isinstance(d_robust, RobustPIDDetector)

        d_multi = get_detector("multimetric")
        assert isinstance(d_multi, MultiMetricPIDDetector)

    def test_detector_factory_unknown_raises(self):
        with pytest.raises(ValueError):
            get_detector("non_existent_detector")
