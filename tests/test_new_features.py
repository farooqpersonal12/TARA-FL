import torch
import numpy as np

from data.dataset import load_mnist, create_clients, create_clients_noniid
from attacks.gradient_poison import GradientScaleAttack, SignFlipAttack
from server.server import Server
from clients.client import Client


def test_server_detector_selection():
    """Verify Server properly initializes standard and robust PID detectors."""
    server_std = Server(detector_type="standard")
    assert server_std.detector_type == "standard"
    assert server_std.detector.__class__.__name__ == "PIDDetector"

    server_rob = Server(detector_type="robust")
    assert server_rob.detector_type == "robust"
    assert server_rob.detector.__class__.__name__ == "RobustPIDDetector"


def test_create_clients_iid_remainder():
    """Verify IID splitting handles non-divisible numbers of clients."""
    dummy_data = torch.utils.data.TensorDataset(
        torch.zeros(65, 1, 28, 28),
        torch.zeros(65, dtype=torch.long)
    )
    splits = create_clients(dummy_data, num_clients=7)
    assert len(splits) == 7
    total_samples = sum(len(s) for s in splits)
    assert total_samples == 65
    sizes = [len(s) for s in splits]
    # 65 = 7 * 9 + 2, so sizes should be [10, 10, 9, 9, 9, 9, 9]
    assert sizes == [10, 10, 9, 9, 9, 9, 9]


def test_create_clients_noniid():
    """Verify Dirichlet non-IID splitting produces valid subsets."""
    dummy_data = torch.utils.data.TensorDataset(
        torch.zeros(100, 1, 28, 28),
        torch.tensor([i % 10 for i in range(100)], dtype=torch.long)
    )
    splits = create_clients_noniid(dummy_data, num_clients=5, alpha=0.5, seed=42)
    assert len(splits) == 5
    total_samples = sum(len(s) for s in splits)
    assert total_samples == 100


def test_gradient_scale_attack():
    """Verify GradientScaleAttack correctly scales parameter update deltas."""
    attack = GradientScaleAttack(scale_factor=10.0)
    update = {
        "weight": torch.tensor([1.0, -2.0, 0.5]),
        "bias": torch.tensor([0.1, -0.2])
    }
    poisoned = attack.apply(update)
    assert torch.allclose(poisoned["weight"], torch.tensor([10.0, -20.0, 5.0]))
    assert torch.allclose(poisoned["bias"], torch.tensor([1.0, -2.0]))


def test_sign_flip_attack():
    """Verify SignFlipAttack correctly reverses parameter update deltas."""
    attack = SignFlipAttack(scale_factor=2.0)
    update = {
        "weight": torch.tensor([1.0, -2.0, 0.5]),
        "bias": torch.tensor([0.1, -0.2])
    }
    poisoned = attack.apply(update)
    assert torch.allclose(poisoned["weight"], torch.tensor([-2.0, 4.0, -1.0]))
    assert torch.allclose(poisoned["bias"], torch.tensor([-0.2, 0.4]))
