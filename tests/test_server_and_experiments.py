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

    def test_quarantined_client_excluded_from_server_aggregation(self):
        """Verify that a quarantined client's update is completely excluded from aggregation."""
        server = Server(model="mnist", enable_quarantine=True, quarantine_threshold=0.20)
        base_params = {k: v.clone() for k, v in server.global_model.state_dict().items()}

        # 3 Honest clients (small clean offset) + 1 Malicious client (large poisoned offset)
        p1 = {k: v.clone() + 0.01 for k, v in base_params.items()}
        p2 = {k: v.clone() + 0.02 for k, v in base_params.items()}
        p3 = {k: v.clone() + 0.03 for k, v in base_params.items()}
        p4 = {k: v.clone() + 1000.0 for k, v in base_params.items()}  # Poisoned

        client_params = [p1, p2, p3, p4]
        client_sizes = [100, 100, 100, 100]
        trust_scores = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.05}  # Client 4 trust is 0.05 < 0.20

        # Evaluate trust into quarantine manager
        server.trust_engine.quarantine_manager.evaluate_client(1, 0.95, 1.0)
        server.trust_engine.quarantine_manager.evaluate_client(2, 0.90, 1.0)
        server.trust_engine.quarantine_manager.evaluate_client(3, 0.85, 1.0)
        server.trust_engine.quarantine_manager.evaluate_client(4, 0.05, 10.0)

        assert server.trust_engine.quarantine_manager.is_quarantined(4)
        assert not server.trust_engine.quarantine_manager.is_quarantined(1)

        new_params, selected_agg = server.aggregate(
            client_parameters=client_params,
            client_sizes=client_sizes,
            trust_scores=trust_scores,
            risk_level="LOW"
        )

        # Output weight must be in range of honest clients [0.01, 0.03] offset, NEVER ~1000.0
        for name in base_params:
            diff = (new_params[name] - base_params[name]).abs().max().item()
            assert diff < 0.1, f"Quarantined client 4 poisoned layer {name} with diff={diff}"

    def test_medium_trust_reduced_influence(self):
        """Verify medium trust client is included but has reduced influence relative to high trust."""
        server = Server(model="mnist", enable_quarantine=True, quarantine_threshold=0.20)
        base_params = {k: v.clone() for k, v in server.global_model.state_dict().items()}

        p1 = {k: v.clone() + 10.0 for k, v in base_params.items()}  # Client 1: High trust (1.0)
        p2 = {k: v.clone() - 10.0 for k, v in base_params.items()}  # Client 2: Medium trust (0.5)

        client_params = [p1, p2]
        client_sizes = [100, 100]
        trust_scores = {1: 1.0, 2: 0.50}

        server.trust_engine.quarantine_manager.evaluate_client(1, 1.0, 1.0)
        server.trust_engine.quarantine_manager.evaluate_client(2, 0.50, 1.2)

        new_params, _ = server.aggregate(
            client_parameters=client_params,
            client_sizes=client_sizes,
            trust_scores=trust_scores,
            risk_level="LOW"
        )

        # Expected weight for client 1: 1.0 / 1.5 = 2/3; client 2: 0.5 / 1.5 = 1/3
        # Expected offset: 2/3 * (+10) + 1/3 * (-10) = +3.333
        for name in base_params:
            offset = (new_params[name] - base_params[name]).mean().item()
            assert abs(offset - 3.3333) < 1e-3, f"Expected offset ~3.333, got {offset}"

    def test_probation_and_recovery_in_server_pipeline(self):
        """Test full multi-round probation and graduation lifecycle in Server pipeline."""
        server = Server(model="mnist", enable_quarantine=True, quarantine_threshold=0.50, required_clean_rounds=2)
        base_params = {k: v.clone() for k, v in server.global_model.state_dict().items()}

        p_honest = {k: v.clone() + 1.0 for k, v in base_params.items()}
        p_reformed = {k: v.clone() + 5.0 for k, v in base_params.items()}

        # Round 1: Client 2 attacks -> quarantined (trust 0.44 drops below 0.50)
        all_pids_r1 = [0.1, 0.1, 0.1, 0.1, 15.0]
        res1 = server.trust_engine.calculate_trust(client_id=1, pid_score=0.1, all_pid_scores=all_pids_r1)
        res2 = server.trust_engine.calculate_trust(client_id=2, pid_score=15.0, all_pid_scores=all_pids_r1)
        assert server.trust_engine.quarantine_manager.is_quarantined(2)

        # Aggregation in Round 1: Client 2 must be excluded (offset should be exactly 1.0 from honest client 1)
        new_params_r1, _ = server.aggregate(
            [p_honest, p_reformed], [100, 100], {1: res1["trust"], 2: res2["trust"]}, risk_level="LOW"
        )
        for name in base_params:
            offset1 = (new_params_r1[name] - base_params[name]).mean().item()
            assert abs(offset1 - 1.0) < 1e-4

        # Round 2: Client 2 behaves cleanly (probation clean round 1)
        all_pids_clean = [0.1, 0.1, 0.1, 0.1, 0.1]
        res1 = server.trust_engine.calculate_trust(client_id=1, pid_score=0.1, all_pid_scores=all_pids_clean)
        res2 = server.trust_engine.calculate_trust(client_id=2, pid_score=0.1, all_pid_scores=all_pids_clean)
        # Still in probation (needs 2 clean rounds)
        assert server.trust_engine.quarantine_manager.is_quarantined(2)
        new_params_r2, _ = server.aggregate(
            [p_honest, p_reformed], [100, 100], {1: res1["trust"], 2: res2["trust"]}, risk_level="LOW"
        )
        for name in base_params:
            offset2 = (new_params_r2[name] - base_params[name]).mean().item()
            assert abs(offset2 - 1.0) < 1e-4

        # Round 3: Client 2 behaves cleanly again (probation clean round 2 -> GRADUATES!)
        res1 = server.trust_engine.calculate_trust(client_id=1, pid_score=0.1, all_pid_scores=all_pids_clean)
        res2 = server.trust_engine.calculate_trust(client_id=2, pid_score=0.1, all_pid_scores=all_pids_clean)
        assert not server.trust_engine.quarantine_manager.is_quarantined(2)
        assert server.trust_engine.quarantine_manager.is_eligible_for_aggregation(2)

        # Aggregation in Round 3: Client 2 is now included!
        new_params_r3, _ = server.aggregate(
            [p_honest, p_reformed], [100, 100], {1: res1["trust"], 2: res2["trust"]}, risk_level="LOW"
        )
        for name in base_params:
            offset3 = (new_params_r3[name] - base_params[name]).mean().item()
            assert offset3 > 1.0  # Reformed client 2 contributed its +5.0 weight!

    def test_all_clients_quarantined_graceful_handling(self):
        """Verify server does not crash when all clients are quarantined."""
        server = Server(model="mnist", enable_quarantine=True, quarantine_threshold=0.20)
        base_params = {k: v.clone() for k, v in server.global_model.state_dict().items()}

        p1 = {k: v.clone() + 10.0 for k, v in base_params.items()}
        p2 = {k: v.clone() + 20.0 for k, v in base_params.items()}

        # Both clients are quarantined
        server.trust_engine.quarantine_manager.evaluate_client(1, 0.05, 5.0)
        server.trust_engine.quarantine_manager.evaluate_client(2, 0.05, 5.0)

        new_params, agg_name = server.aggregate(
            client_parameters=[p1, p2],
            client_sizes=[100, 100],
            trust_scores={1: 0.05, 2: 0.05},
            risk_level="HIGH"
        )

        assert agg_name == "QUARANTINE_ALL_RETAINED"
        for name in base_params:
            assert torch.equal(new_params[name], base_params[name])

    def test_single_eligible_client_aggregation(self):
        """Verify server and aggregators handle exactly 1 eligible client cleanly."""
        server = Server(model="mnist", enable_quarantine=True, quarantine_threshold=0.20)
        base_params = {k: v.clone() for k, v in server.global_model.state_dict().items()}

        p1 = {k: v.clone() + 2.5 for k, v in base_params.items()}
        p2 = {k: v.clone() + 100.0 for k, v in base_params.items()}

        # Client 2 quarantined, Client 1 eligible
        server.trust_engine.quarantine_manager.evaluate_client(1, 0.90, 1.0)
        server.trust_engine.quarantine_manager.evaluate_client(2, 0.05, 10.0)

        new_params, _ = server.aggregate(
            client_parameters=[p1, p2],
            client_sizes=[100, 100],
            trust_scores={1: 0.90, 2: 0.05},
            risk_level="MEDIUM"
        )

        for name in base_params:
            diff = (new_params[name] - p1[name]).abs().max().item()
            assert diff < 1e-5, "Single eligible client weights should be preserved exactly."

    def test_four_methods_reproducibility(self, monkeypatch):
        """Verify that all four comparison methods execute deterministically given a fixed seed."""
        import experiments.run_unified as ru
        from argparse import Namespace

        dummy_train = TensorDataset(torch.randn(30, 1, 28, 28), torch.randint(0, 10, (30,)))
        dummy_test = TensorDataset(torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,)))
        monkeypatch.setattr(ru, "load_mnist", lambda: (dummy_train, dummy_test))

        cfg = Namespace(
            method="fedavg",
            num_clients=3,
            rounds=2,
            local_epochs=1,
            malicious_ratio=0.33,
            attack="label_flip",
            flip_ratio=0.5,
            scale_factor=10.0,
            data_split="iid",
            alpha=0.5,
            detector_type="standard"
        )

        for method in ["fedavg", "trust_fedavg", "trust_robust", "tara"]:
            # Run 1
            res1 = ru.run_experiment(cfg, seed=42, method_override=method)
            # Run 2 with same seed
            res2 = ru.run_experiment(cfg, seed=42, method_override=method)

            assert len(res1) == len(res2) == 2
            for r1, r2 in zip(res1, res2):
                assert r1["accuracy"] == pytest.approx(r2["accuracy"], abs=1e-6)
                assert r1["loss"] == pytest.approx(r2["loss"], abs=1e-5)
                assert r1["aggregator"] == r2["aggregator"]

    def test_controlled_comparison_fairness(self):
        """Verify that all four methods share identical data partitions and initial weights for the same seed."""
        from data.dataset import load_mnist, create_clients
        import random

        train_ds, _ = load_mnist()

        # Seed 100 partitions
        random.seed(100)
        torch.manual_seed(100)
        split_a = create_clients(train_ds, num_clients=4)

        random.seed(100)
        torch.manual_seed(100)
        split_b = create_clients(train_ds, num_clients=4)

        assert len(split_a) == len(split_b) == 4
        for i in range(4):
            assert len(split_a[i]) == len(split_b[i])
