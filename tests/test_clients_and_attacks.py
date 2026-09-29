import pytest
import torch
from torch.utils.data import TensorDataset

from clients.client import Client
from model.model import MNISTModel, CIFAR10ConvNet, MLPModel
from data.dataset import (
    create_clients,
    create_clients_noniid,
    create_clients_pathological,
    sample_active_clients,
    simulate_client_dropout
)
from attacks.label_flip import LabelFlipDataset, LabelFlipAttack
from attacks.gradient_poison import GradientScaleAttack, SignFlipAttack
from attacks.backdoor import BackdoorDataset, BackdoorAttack
from attacks.gaussian_noise import GaussianNoiseAttack, RandomParameterZeroAttack
from attacks.sybil_collusion import SybilCollusionCoordinator
from attacks.attack_factory import AttackFactory, get_attack


@pytest.fixture
def dummy_mnist_dataset():
    images = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    return TensorDataset(images, labels)


@pytest.fixture
def dummy_cifar_dataset():
    images = torch.randn(60, 3, 32, 32)
    labels = torch.randint(0, 10, (60,))
    return TensorDataset(images, labels)


class TestClientSuite:
    """Test suite for Client functionality."""

    def test_client_init_and_train_mnist(self, dummy_mnist_dataset):
        client = Client(client_id=1, dataset=dummy_mnist_dataset, model="mnist", lr=0.05)
        initial_params = client.get_parameters()
        client.train(epochs=1)
        updated_params = client.get_parameters()

        # Check weights updated
        changed = False
        for k in initial_params:
            if not torch.equal(initial_params[k], updated_params[k]):
                changed = True
                break
        assert changed

    def test_client_optimizers(self, dummy_mnist_dataset):
        # Adam
        c_adam = Client(client_id=1, dataset=dummy_mnist_dataset, optimizer_type="adam")
        c_adam.train(epochs=1)

        # AdamW
        c_adamw = Client(client_id=2, dataset=dummy_mnist_dataset, optimizer_type="adamw", weight_decay=1e-4)
        c_adamw.train(epochs=1)

    def test_client_with_cifar_model(self, dummy_cifar_dataset):
        client = Client(client_id=1, dataset=dummy_cifar_dataset, model="cifar10")
        client.train(epochs=1)
        loss, acc = client.evaluate()
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_client_differential_privacy(self, dummy_mnist_dataset):
        client_dp = Client(
            client_id=1,
            dataset=dummy_mnist_dataset,
            enable_dp=True,
            dp_clip_norm=1.0,
            dp_noise_multiplier=0.05
        )
        client_dp.train(epochs=1)
        params = client_dp.get_parameters()
        assert len(params) > 0

    def test_client_get_update(self, dummy_mnist_dataset):
        model = MNISTModel()
        global_params = model.state_dict()
        client = Client(client_id=1, dataset=dummy_mnist_dataset, model=model)
        client.train(epochs=1)
        update = client.get_update(global_params)

        assert len(update) == len(global_params)
        for k in update:
            assert update[k].shape == global_params[k].shape


class TestDataPartitioning:
    """Test suite for IID and Non-IID data partitioning."""

    def test_iid_partitioning(self, dummy_mnist_dataset):
        clients = create_clients(dummy_mnist_dataset, num_clients=4)
        assert len(clients) == 4
        total_len = sum(len(c) for c in clients)
        assert total_len == 100

    def test_noniid_dirichlet(self, dummy_mnist_dataset):
        clients = create_clients_noniid(dummy_mnist_dataset, num_clients=3, alpha=0.5)
        assert len(clients) == 3
        total_len = sum(len(c) for c in clients)
        assert total_len == 100

    def test_pathological_noniid(self, dummy_mnist_dataset):
        clients = create_clients_pathological(dummy_mnist_dataset, num_clients=5, classes_per_client=2)
        assert len(clients) == 5
        total_len = sum(len(c) for c in clients)
        assert total_len == 100

    def test_client_sampling_and_dropout(self):
        client_ids = list(range(1, 21))
        sampled = sample_active_clients(client_ids, sample_ratio=0.5, seed=42)
        assert len(sampled) == 10

        surviving = simulate_client_dropout(client_ids, dropout_rate=0.2, seed=42)
        assert len(surviving) < 20
        assert len(surviving) > 0


class TestAttackSuite:
    """Test suite for adversarial attack implementations."""

    def test_label_flip_attack(self, dummy_mnist_dataset):
        # Target map flip
        target_map = {0: 9, 1: 8}
        attack = LabelFlipAttack(flip_ratio=1.0, target_map=target_map)
        poisoned_ds = attack.apply(dummy_mnist_dataset)

        for i in range(len(dummy_mnist_dataset)):
            orig_img, orig_lbl = dummy_mnist_dataset[i]
            p_img, p_lbl = poisoned_ds[i]
            if orig_lbl.item() in target_map:
                assert p_lbl == target_map[orig_lbl.item()]

    def test_gradient_scale_and_sign_flip(self):
        update = {"w": torch.tensor([1.0, -2.0, 3.0]), "b": torch.tensor([0.5])}

        # Scale attack
        scale_attack = GradientScaleAttack(scale_factor=5.0)
        scaled = scale_attack.apply(update)
        assert torch.equal(scaled["w"], torch.tensor([5.0, -10.0, 15.0]))

        # Sign flip attack
        sign_attack = SignFlipAttack(scale_factor=2.0)
        flipped = sign_attack.apply(update)
        assert torch.equal(flipped["w"], torch.tensor([-2.0, 4.0, -6.0]))

    def test_backdoor_attack(self, dummy_mnist_dataset):
        attack = BackdoorAttack(poison_ratio=1.0, target_label=7, trigger_size=3)
        poisoned_ds = attack.apply(dummy_mnist_dataset)

        img, label = poisoned_ds[0]
        assert label == 7
        # Trigger pixel check on bottom right corner
        assert img[0, -1, -1] == 1.0

    def test_gaussian_noise_and_param_zero_attack(self):
        update = {"w": torch.ones(10, 10), "b": torch.zeros(5)}

        noise_attack = GaussianNoiseAttack(std=0.5)
        noisy = noise_attack.apply(update)
        assert not torch.equal(noisy["w"], update["w"])

        zero_attack = RandomParameterZeroAttack(zero_ratio=0.5)
        zeroed = zero_attack.apply(update)
        assert (zeroed["w"] == 0).sum().item() > 0

    def test_sybil_collusion(self):
        client_updates = {
            1: {"w": torch.tensor([1.0, 1.0])},
            2: {"w": torch.tensor([2.0, 2.0])},  # Sybil 1
            3: {"w": torch.tensor([3.0, 3.0])},  # Sybil 2
        }
        coordinator = SybilCollusionCoordinator(sybil_client_ids=[2, 3], scale_factor=4.0)
        coordinated = coordinator.coordinate_updates(client_updates, leader_id=2)

        # Sybil 2 and Sybil 3 now have identical scaled updates
        assert torch.equal(coordinated[2]["w"], torch.tensor([8.0, 8.0]))
        assert torch.equal(coordinated[3]["w"], torch.tensor([8.0, 8.0]))
        # Client 1 untouched
        assert torch.equal(coordinated[1]["w"], torch.tensor([1.0, 1.0]))

    def test_attack_factory(self):
        attacks = AttackFactory.list_attacks()
        assert "label_flip" in attacks
        assert "gradient_scale" in attacks
        assert "backdoor" in attacks
        assert "gaussian_noise" in attacks

        att = get_attack("gradient_scale", scale_factor=8.0)
        assert isinstance(att, GradientScaleAttack)
        assert att.scale_factor == 8.0
