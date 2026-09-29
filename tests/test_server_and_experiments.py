import os
import tempfile
import pytest
import torch
from torch.utils.data import TensorDataset

from server.server import Server
from model.model import MNISTModel, CIFAR10ConvNet
from experiments.experiment_config import ExperimentConfig


@pytest.fixture
def dummy_test_dataset():
    images = torch.randn(50, 1, 28, 28)
    labels = torch.randint(0, 10, (50,))
    return TensorDataset(images, labels)


class TestServerAndExperimentsSuite:
    """Test suite for Server orchestrator and ExperimentConfig."""

    def test_server_init_dynamic_model_and_detector(self):
        # Default MNIST + Robust
        s_default = Server()
        assert isinstance(s_default.global_model, MNISTModel)

        # CIFAR-10 model + MultiMetric detector
        s_cifar = Server(model="cifar10", detector_type="multimetric")
        assert isinstance(s_cifar.global_model, CIFAR10ConvNet)

    def test_server_evaluate_accuracy_and_loss(self, dummy_test_dataset):
        server = Server(model="mnist")

        # Standard accuracy return
        acc = server.evaluate(dummy_test_dataset, return_loss=False)
        assert 0.0 <= acc <= 1.0

        # Loss and accuracy return
        loss, acc = server.evaluate(dummy_test_dataset, return_loss=True)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_server_checkpoint_save_and_load(self, dummy_test_dataset):
        server = Server(model="mnist")
        server.trust_engine.history.update(1, trust_score=0.92, anomaly_score=1.0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            server.save_checkpoint(temp_path, round_number=7, extra_meta={"exp_id": "test_exp"})

            new_server = Server(model="mnist")
            loaded_round = new_server.load_checkpoint(temp_path)

            assert loaded_round == 7
            assert new_server.trust_engine.history.get_latest_trust(1) == 0.92
            for k in server.global_model.state_dict():
                assert torch.equal(server.global_model.state_dict()[k], new_server.global_model.state_dict()[k])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_experiment_config_defaults(self):
        config = ExperimentConfig(
            experiment_name="cifar_robust_test",
            dataset_name="cifar10",
            model_name="cifar10",
            num_clients=5,
            num_malicious=1,
            attack_type="gradient_scale",
            seeds=[42, 100]
        )
        assert config.num_clients == 5
        assert len(config.seeds) == 2
        assert config.attack_type == "gradient_scale"
