"""
TARA-FL QA Phase 3: Comprehensive Module-by-Module Functional Validation
Tests every module independently with strict assertions and mathematical verification.
"""

import sys
import os
import json
import copy
import math
import traceback
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.base_model import BaseFLModel
from model.model import (
    MNISTModel, FashionMNISTModel, CIFAR10ConvNet, ResNet18Small, MLPModel,
    ModelFactory, get_model, list_available_models
)
from clients.client import Client
from data.dataset import (
    create_clients, create_clients_noniid, create_clients_pathological,
    sample_active_clients, simulate_client_dropout, load_dataset
)
from attacks.base_attack import BaseAttack
from attacks.label_flip import LabelFlipDataset, LabelFlipAttack
from attacks.gradient_poison import GradientScaleAttack, SignFlipAttack
from attacks.backdoor import BackdoorDataset, BackdoorAttack
from attacks.gaussian_noise import GaussianNoiseAttack, RandomParameterZeroAttack
from attacks.sybil_collusion import SybilCollusionCoordinator
from attacks.attack_factory import AttackFactory, get_attack
from detection.base_detector import BaseDetector
from detection.pid_detector import PIDDetector
from detection.robust_pid_detector import RobustPIDDetector
from detection.multimetric_detector import MultiMetricPIDDetector
from detection.detector_factory import DetectorFactory, get_detector
from trustengine.trust_engine import TrustEngine
from trustengine.trust_history import TrustHistory
from trustengine.quarantine import QuarantineManager, TrustZone
from trustengine.probe_evaluator import ValidationProbeEvaluator
from risk.round_risk import RoundRisk
from risk.threat_classifier import ThreatClassifier, ThreatType
from aggregation.base_aggregator import BaseAggregator
from aggregation.strategies import (
    TrustAwareFedAvg, TrustWeightedRobustTrimming, TrustWeightedMedian,
    TrustAwareFedAvgM, MultiKrumAggregator
)
from aggregation.candidate_selector import CandidateModelSelector
from aggregation.adaptive_aggregator import AdaptiveAggregator
from aggregation.aggregator_factory import AggregatorFactory, get_aggregator
from server.server import Server


def run_test_section(section_name, test_fn):
    print(f"\n{'='*60}\n>>> STARTING: {section_name}\n{'='*60}")
    try:
        results = test_fn()
        print(f">>> [PASS] {section_name}")
        return {"status": "PASS", "details": results, "error": None}
    except Exception as e:
        exc_str = traceback.format_exc()
        print(f">>> [FAIL] {section_name}\nError: {e}\n{exc_str}")
        return {"status": "FAIL", "details": None, "error": exc_str}


# ==============================================================================
# A. MODEL LAYER TESTS
# ==============================================================================
def test_model_layer():
    results = {}
    models_to_test = {
        "mnist": ((4, 1, 28, 28), (4, 10)),
        "fashion_mnist": ((4, 1, 28, 28), (4, 10)),
        "cifar10": ((4, 3, 32, 32), (4, 10)),
        "resnet18": ((4, 3, 32, 32), (4, 10)),
        "mlp": ((4, 784), (4, 10))
    }

    avail = list_available_models()
    assert set(models_to_test.keys()).issubset(set(avail)), f"Model registry mismatch: {avail}"

    for model_name, (in_shape, out_shape) in models_to_test.items():
        m = get_model(model_name)
        assert isinstance(m, BaseFLModel), f"{model_name} is not BaseFLModel"
        
        # Test forward
        x = torch.randn(*in_shape)
        out = m(x)
        assert out.shape == out_shape, f"{model_name} out shape {out.shape} != {out_shape}"
        
        # Test parameter vector roundtrip
        vec = m.get_parameter_vector()
        assert isinstance(vec, torch.Tensor) and vec.ndim == 1
        num_params = m.count_parameters()
        assert len(vec) == num_params, f"{model_name} param vec length {len(vec)} != count {num_params}"
        
        # Create fresh model and restore
        m_fresh = get_model(model_name)
        m_fresh.set_parameter_vector(vec)
        vec_restored = m_fresh.get_parameter_vector()
        diff = torch.norm(vec - vec_restored).item()
        assert diff < 1e-6, f"{model_name} parameter roundtrip norm diff {diff} >= 1e-6"
        
        # Test weights dict roundtrip
        w_dict = m.get_weights()
        m_fresh.set_weights(w_dict)
        for k, v in m.get_weights().items():
            assert torch.equal(v, m_fresh.get_weights()[k]), f"Weight mismatch for {k}"
            
        # Test gradient extraction
        loss = out.sum()
        m.zero_grad()
        loss.backward()
        g_vec = m.get_gradient_vector()
        assert g_vec is not None and len(g_vec) == num_params
        
        # Test checkpoint save / load
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            m.save_checkpoint(tmp_path)
            m_loaded = get_model(model_name)
            m_loaded.load_checkpoint(tmp_path)
            diff_chk = torch.norm(m.get_parameter_vector() - m_loaded.get_parameter_vector()).item()
            assert diff_chk < 1e-6, f"{model_name} checkpoint restore diff {diff_chk}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        results[model_name] = {"num_params": num_params, "diff": diff}

    return results


# ==============================================================================
# B. CLIENT LAYER TESTS
# ==============================================================================
def test_client_layer():
    results = {}
    x = torch.randn(80, 1, 28, 28)
    y = torch.randint(0, 10, (80,))
    ds = TensorDataset(x, y)

    # 1. Optimizers
    for opt_name in ["sgd", "adam", "adamw"]:
        client = Client(client_id=1, dataset=ds, model="mnist", lr=0.01, optimizer_type=opt_name)
        w_before = client.model.get_parameter_vector().clone()
        res = client.train(epochs=1)
        w_after = client.model.get_parameter_vector()
        delta_norm = torch.norm(w_after - w_before).item()
        assert delta_norm > 0.0, f"Optimizer {opt_name} did not update weights! delta_norm={delta_norm}"
        results[f"optimizer_{opt_name}_delta"] = delta_norm

    # 2. Local evaluation
    eval_loss, eval_acc = client.evaluate()
    assert isinstance(eval_loss, float) and not math.isnan(eval_loss)
    assert 0.0 <= eval_acc <= 1.0

    # 3. Model injection
    custom_model = MLPModel(input_dim=784, hidden_dims=[64], num_classes=10)
    x_mlp = torch.randn(80, 784)
    ds_mlp = TensorDataset(x_mlp, y)
    client_mlp = Client(client_id=2, dataset=ds_mlp, model=custom_model, lr=0.01)
    client_mlp.train(epochs=1)
    assert client_mlp.model.count_parameters() == custom_model.count_parameters()

    # 4. LDP (Gradient clipping & Gaussian noise)
    torch.manual_seed(42)
    client_dp_nonoise = Client(client_id=3, dataset=ds, model="mnist", enable_dp=True, dp_clip_norm=0.5, dp_noise_multiplier=0.0)
    client_dp_noisy = Client(client_id=4, dataset=ds, model="mnist", enable_dp=True, dp_clip_norm=0.5, dp_noise_multiplier=0.1)
    
    # Set identical starting weights
    w_init = get_model("mnist").get_weights()
    client_dp_nonoise.set_model(w_init)
    client_dp_noisy.set_model(w_init)
    
    torch.manual_seed(100)
    client_dp_nonoise.train(epochs=1)
    torch.manual_seed(100)
    client_dp_noisy.train(epochs=1)
    
    diff_dp = torch.norm(client_dp_nonoise.model.get_parameter_vector() - client_dp_noisy.model.get_parameter_vector()).item()
    assert diff_dp > 0.0, "LDP noise did not perturb weights!"
    results["ldp_perturbation_norm"] = diff_dp

    return results


# ==============================================================================
# C. DATA PARTITIONING TESTS
# ==============================================================================
def test_data_partitioning():
    results = {}
    num_samples = 300
    x = torch.randn(num_samples, 1, 28, 28)
    y = torch.tensor([i % 10 for i in range(num_samples)])
    ds = TensorDataset(x, y)

    # 1. IID Partitioning
    num_clients = 5
    iid_clients = create_clients(ds, num_clients=num_clients)
    assert len(iid_clients) == num_clients
    total_iid = sum(len(c) for c in iid_clients)
    assert total_iid == num_samples, f"IID sample count {total_iid} != {num_samples}"

    # 2. Dirichlet Non-IID
    dir_clients = create_clients_noniid(ds, num_clients=num_clients, alpha=0.1, seed=42)
    assert len(dir_clients) == num_clients
    total_dir = sum(len(c) for c in dir_clients)
    assert total_dir == num_samples, f"Dirichlet sample count {total_dir} != {num_samples}"
    
    # Check class distributions vary across clients
    client_class_counts = []
    for c in dir_clients:
        labels = [y[idx].item() for idx in c.indices]
        counts = [labels.count(k) for k in range(10)]
        client_class_counts.append(counts)
    std_across_clients = np.std(client_class_counts, axis=0).mean()
    assert std_across_clients > 0.0, "Dirichlet partition is completely uniform!"
    results["dirichlet_std"] = float(std_across_clients)

    # 3. Pathological Non-IID
    path_clients = create_clients_pathological(ds, num_clients=num_clients, classes_per_client=2, seed=42)
    for i, c in enumerate(path_clients):
        labels = set(y[idx].item() for idx in c.indices)
        assert len(labels) <= 2, f"Client {i} has {len(labels)} classes > 2 in pathological partition!"

    # 4. Client sampling & Dropout
    active_sampled = sample_active_clients(list(range(20)), sample_ratio=0.5, seed=42)
    assert len(active_sampled) == 10
    
    dropout_survived = simulate_client_dropout(list(range(20)), dropout_rate=0.3, seed=42)
    assert 0 < len(dropout_survived) <= 20
    results["dropout_survivors"] = len(dropout_survived)

    return results


# ==============================================================================
# D. ATTACK SUITE TESTS
# ==============================================================================
def test_attack_suite():
    results = {}
    x = torch.randn(50, 1, 28, 28)
    y = torch.randint(0, 10, (50,))
    ds = TensorDataset(x, y)
    
    # 1. Label Flip
    lf_ds = LabelFlipDataset(ds, flip_ratio=1.0, num_classes=10, shift=1, seed=42)
    for i in range(len(ds)):
        orig_y = int(ds[i][1].item())
        flipped_y = int(lf_ds[i][1].item())
        assert flipped_y == (orig_y + 1) % 10

    # 2. Gradient Scaling
    gs = get_attack("gradient_scale", scale_factor=10.0)
    update = {"w": torch.tensor([1.0, 2.0, -3.0])}
    scaled = gs.apply(update)
    assert torch.allclose(scaled["w"], torch.tensor([10.0, 20.0, -30.0]))

    # 3. Sign Flip
    sf = get_attack("sign_flip")
    flipped = sf.apply(update)
    assert torch.allclose(flipped["w"], torch.tensor([-1.0, -2.0, 3.0]))

    # 4. Gaussian Noise
    gn = get_attack("gaussian_noise", mean=0.0, std=0.5)
    noisy = gn.apply(update)
    assert not torch.equal(noisy["w"], update["w"])
    assert torch.norm(noisy["w"] - update["w"]).item() > 0.0

    # 5. Parameter Zero-Out
    zo = get_attack("zero_out", zero_ratio=1.0)
    zeroed = zo.apply(update)
    assert torch.allclose(zeroed["w"], torch.zeros(3))


    # 6. Backdoor Attack
    bd = get_attack("backdoor", poison_ratio=1.0, target_label=7, trigger_size=3)
    bd_ds = bd.apply(ds)
    assert len(bd_ds) == len(ds)
    img, lbl = bd_ds[0]
    assert lbl == 7
    # Trigger is at bottom-right
    h, w = img.shape[1], img.shape[2]
    assert torch.all(img[:, h-3:h, w-3:w] == 1.0)


    # 7. Sybil Collusion
    sybil = SybilCollusionCoordinator(sybil_client_ids=[10, 11], scale_factor=5.0)
    u10 = {"w": torch.tensor([1.0, 1.0])}
    u11 = {"w": torch.tensor([2.0, 2.0])}
    all_ups = {10: u10, 11: u11, 1: {"w": torch.tensor([0.1, 0.1])}}
    colluded = sybil.coordinate_updates(all_ups, leader_id=10)
    # Both 10 and 11 should receive u10 * 5.0 = [5.0, 5.0]
    assert torch.allclose(colluded[10]["w"], torch.tensor([5.0, 5.0]))
    assert torch.allclose(colluded[11]["w"], torch.tensor([5.0, 5.0]))
    # Honest client 1 should be unchanged
    assert torch.allclose(colluded[1]["w"], torch.tensor([0.1, 0.1]))


    return {"status": "all 7 attacks verified"}


# ==============================================================================
# E. DETECTION ENGINE TESTS
# ==============================================================================
def test_detection_engine():
    results = {}
    base_m = MNISTModel()
    honest_updates = {
        1: {k: torch.randn_like(v) * 0.005 for k, v in base_m.get_weights().items()},
        2: {k: torch.randn_like(v) * 0.005 for k, v in base_m.get_weights().items()},
        3: {k: torch.randn_like(v) * 0.005 for k, v in base_m.get_weights().items()},
        4: {k: torch.randn_like(v) * 0.005 for k, v in base_m.get_weights().items()}
    }
    # Client 5 is malicious with huge magnitude
    malicious_update = {k: v * 50.0 for k, v in honest_updates[1].items()}
    updates = dict(honest_updates)
    updates[5] = malicious_update

    # 1. Standard PID Detector
    pid = get_detector("pid", kp=1.0, ki=0.2, kd=0.1)
    dists_pid, scores_pid = pid.calculate_scores(updates)
    assert scores_pid[5] > max(scores_pid[1], scores_pid[2], scores_pid[3], scores_pid[4])

    # 2. Robust PID Detector
    r_pid = get_detector("robust_pid", kp=1.0, ki=0.2, kd=0.1)
    dists_rpid, scores_rpid = r_pid.calculate_scores(updates)
    assert scores_rpid[5] > max(scores_rpid[1], scores_rpid[2], scores_rpid[3], scores_rpid[4])

    # 3. MultiMetric PID Detector
    mm_pid = get_detector("multimetric", weight_euclidean=0.4, weight_cosine=0.3, weight_layerwise=0.3)
    dists_mm, scores_mm = mm_pid.calculate_scores(updates)
    assert scores_mm[5] > max(scores_mm[1], scores_mm[2], scores_mm[3], scores_mm[4])
    
    # 4. Anti-windup and sliding window verification
    for r in range(2, 10):
        _, s = mm_pid.calculate_scores(updates)
        assert not math.isinf(s[5]) and not math.isnan(s[5])

    results["pid_malicious_ratio"] = scores_pid[5] / max(scores_pid[1], 1e-6)
    results["multimetric_malicious_ratio"] = scores_mm[5] / max(scores_mm[1], 1e-6)
    return results


# ==============================================================================
# F. TRUST ENGINE & QUARANTINE TESTS
# ==============================================================================
def test_trust_engine_and_quarantine():
    results = {}
    engine = TrustEngine(decay_factor=0.8, quarantine_threshold=0.20, enable_quarantine=True)
    
    # Simulate honest rounds vs malicious rounds
    for r in range(1, 6):
        # Client 1: honest (low anomaly 0.05)
        # Client 2: malicious (high anomaly 5.0)
        all_pid = [0.05, 5.0]
        res1 = engine.calculate_trust(client_id=1, pid_score=0.05, all_pid_scores=all_pid)
        res2 = engine.calculate_trust(client_id=2, pid_score=5.0, all_pid_scores=all_pid)

    t1 = engine.history.get_latest_trust(1)
    t2 = engine.history.get_latest_trust(2)
    assert t1 > t2, f"Honest trust {t1} not > malicious trust {t2}"

    # Quarantine Manager tests
    qm = QuarantineManager(quarantine_threshold=0.20, required_clean_rounds=2, max_quarantine_rounds=5, enable_eviction=True)
    
    # Test Quarantine entry
    zone_mal = qm.evaluate_client(client_id=10, trust_score=0.10, relative_anomaly=4.0)
    assert zone_mal == TrustZone.QUARANTINE
    assert qm.is_quarantined(10)
    assert not qm.is_eligible_for_aggregation(10)

    # Test Probation clean rounds
    qm.evaluate_client(client_id=10, trust_score=0.70, relative_anomaly=0.1)
    qm.evaluate_client(client_id=10, trust_score=0.75, relative_anomaly=0.1)
    # After 2 clean rounds, should graduate
    assert not qm.is_quarantined(10)
    assert qm.is_eligible_for_aggregation(10)

    # Test Eviction on persistent anomaly
    for r in range(6):
        qm.evaluate_client(client_id=20, trust_score=0.10, relative_anomaly=5.0)
    assert qm.is_evicted(20)
    assert not qm.is_eligible_for_aggregation(20)

    # Test JSON Persistence
    history = TrustHistory()
    history.update_trust(client_id=1, trust_score=0.95)
    history.update_trust(client_id=1, trust_score=0.92)
    h_dict = history.to_dict()
    h_loaded = TrustHistory()
    h_loaded.from_dict(h_dict)
    assert len(h_loaded.get_trust_history(1)) == 2
    assert h_loaded.get_latest_trust(1) == 0.92

    return {"status": "trust engine, quarantine and persistence passed"}


# ==============================================================================
# G. VALIDATION PROBE TESTS
# ==============================================================================
def test_validation_probe():
    results = {}
    m = MNISTModel()
    x = torch.randn(40, 1, 28, 28)
    y = torch.randint(0, 10, (40,))
    val_ds = TensorDataset(x, y)
    evaluator = ValidationProbeEvaluator(probe_dataset=val_ds, batch_size=20)

    # Baseline loss
    base_loss = evaluator._compute_loss(m)
    assert isinstance(base_loss, float) and not math.isnan(base_loss)

    # Clean small update vs harmful large update
    clean_update = {k: torch.randn_like(v) * 0.001 for k, v in m.get_weights().items()}
    harmful_update = {k: torch.randn_like(v) * 10.0 for k, v in m.get_weights().items()}
    
    updates = {1: clean_update, 2: harmful_update}
    quality_scores = evaluator.evaluate_updates(m, updates)

    # CRITICAL: Verify model was NOT permanently modified by probe evaluation!
    base_loss_after = evaluator._compute_loss(m)
    assert abs(base_loss - base_loss_after) < 1e-6, "Probe evaluator mutated the global model!"
    
    assert quality_scores[1] > quality_scores[2], f"Clean quality {quality_scores[1]} <= harmful {quality_scores[2]}"
    
    results["q_clean"] = quality_scores[1]
    results["q_harm"] = quality_scores[2]
    return results


# ==============================================================================
# H. ROUND RISK & THREAT CLASSIFIER TESTS
# ==============================================================================
def test_round_risk_and_threats():
    results = {}
    rr = RoundRisk(low_risk_threshold=0.3, medium_risk_threshold=0.6)

    # Low risk scenario
    anomalies_low = {1: 0.05, 2: 0.04, 3: 0.06, 4: 0.05}
    trust_low = {1: 0.95, 2: 0.98, 3: 0.92, 4: 0.96}
    risk_low, regime_low, susp_low = rr.calculate_risk(anomalies_low, trust_low)
    assert 0.0 <= risk_low <= 1.0
    assert regime_low == "LOW"

    # High risk scenario
    anomalies_high = {1: 5.0, 2: 4.5, 3: 6.0, 4: 0.05}
    trust_high = {1: 0.10, 2: 0.15, 3: 0.05, 4: 0.95}
    risk_high, regime_high, susp_high = rr.calculate_risk(anomalies_high, trust_high)
    assert 0.0 <= risk_high <= 1.0
    assert risk_high > risk_low
    assert regime_high in ["MEDIUM", "HIGH"]

    # Accuracy degradation escalation
    rr.previous_accuracy = 0.85
    escalated_risk, _, _ = rr.calculate_risk(anomalies_low, trust_low, current_accuracy=0.60)
    assert escalated_risk > risk_low, "Performance drop did not escalate risk!"

    # Threat Classifier
    tc = ThreatClassifier()
    # Magnitude scaling
    t_scale = tc.classify_threat(
        distances={1: 15.0, 2: 0.2},
        trust_scores={1: 0.1, 2: 0.9}
    )
    assert t_scale == ThreatType.MAGNITUDE_SCALING

    # Sign flip / angular (3 honest updates in positive direction, 1 sign-flipped)
    u1 = {"w": torch.tensor([1.0, 1.0])}
    u2 = {"w": torch.tensor([1.0, 1.0])}
    u3 = {"w": torch.tensor([1.0, 1.0])}
    u4 = {"w": torch.tensor([-1.0, -1.0])}
    t_angular = tc.classify_threat(
        distances={1: 0.5, 2: 0.5, 3: 0.5, 4: 2.0},
        trust_scores={1: 0.9, 2: 0.9, 3: 0.9, 4: 0.3},
        client_updates={1: u1, 2: u2, 3: u3, 4: u4}
    )
    assert t_angular == ThreatType.SIGN_FLIP_ANGULAR

    return {"risk_low": risk_low, "risk_high": risk_high, "escalated": escalated_risk}


# ==============================================================================
# I. AGGREGATION SUITE & CANDIDATE SELECTION TESTS
# ==============================================================================
def test_aggregation_suite():
    results = {}
    m = MNISTModel()
    w_base = m.get_weights()
    
    # 3 honest updates, 1 outlier update
    u1 = {k: v + torch.ones_like(v) * 0.1 for k, v in w_base.items()}
    u2 = {k: v + torch.ones_like(v) * 0.12 for k, v in w_base.items()}
    u3 = {k: v + torch.ones_like(v) * 0.08 for k, v in w_base.items()}
    u_mal = {k: v + torch.ones_like(v) * 50.0 for k, v in w_base.items()}
    
    client_params = [u1, u2, u3, u_mal]
    client_sizes = [100, 100, 100, 100]
    trust_scores = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.05}
    client_updates = {
        1: {k: u1[k] - w_base[k] for k in w_base},
        2: {k: u2[k] - w_base[k] for k in w_base},
        3: {k: u3[k] - w_base[k] for k in w_base},
        4: {k: u_mal[k] - w_base[k] for k in w_base},
    }

    # 1. Trust-Aware FedAvg
    fedavg = TrustAwareFedAvg()
    agg_fedavg = fedavg.aggregate(client_params, client_sizes, trust_scores)
    assert set(agg_fedavg.keys()) == set(w_base.keys())

    # 2. Trust-Weighted Robust Trimming
    robust = TrustWeightedRobustTrimming(trim_ratio=0.25)
    agg_robust = robust.aggregate(client_params, client_sizes, trust_scores, client_updates=client_updates)
    diff_rob = (agg_robust["fc2.bias"] - w_base["fc2.bias"]).mean().item()
    assert diff_rob < 1.0, f"Trimming failed to exclude outlier! diff={diff_rob}"

    # 3. Trust-Weighted Median
    median = TrustWeightedMedian()
    agg_med = median.aggregate(client_params, client_sizes, trust_scores)
    diff_med = (agg_med["fc2.bias"] - w_base["fc2.bias"]).mean().item()
    assert diff_med < 1.0, f"Median failed to resist outlier! diff={diff_med}"

    # 4. Trust-Aware FedAvgM (Momentum)
    fedavgm = TrustAwareFedAvgM(beta=0.9)
    agg_m1 = fedavgm.aggregate(client_params, client_sizes, trust_scores, global_parameters=w_base)
    agg_m2 = fedavgm.aggregate(client_params, client_sizes, trust_scores, global_parameters=agg_m1)
    assert len(fedavgm.momentum_buffer) > 0

    # 5. MultiKrum
    krum = MultiKrumAggregator(num_malicious=1, to_select=2)
    agg_krum = krum.aggregate(client_params, client_sizes, trust_scores)
    diff_krum = (agg_krum["fc2.bias"] - w_base["fc2.bias"]).mean().item()
    assert diff_krum < 1.0, f"Krum failed to resist outlier! diff={diff_krum}"

    # 6. Candidate Model Selector
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    val_ds = TensorDataset(x, y)
    selector = CandidateModelSelector(probe_dataset=val_ds)
    best_params, best_name, eval_scores = selector.select_best_candidate(
        model_template=m,
        client_parameters=client_params,
        client_sizes=client_sizes,
        trust_scores=trust_scores,
        client_updates=client_updates
    )
    assert best_name in ["TRUST_AWARE_FEDAVG", "TRUST_WEIGHTED_ROBUST", "TRUST_WEIGHTED_MEDIAN", "MULTI_KRUM"]
    assert best_params is not None

    # 7. Adaptive Aggregator
    adaptive = AdaptiveAggregator()
    agg_adapt, selected_name = adaptive.aggregate(
        client_parameters=client_params,
        client_sizes=client_sizes,
        trust_scores=trust_scores,
        risk_level="HIGH",
        client_updates=client_updates,
        mode="risk_routing"
    )
    assert selected_name == "TRUST_WEIGHTED_MEDIAN"
    assert set(agg_adapt.keys()) == set(w_base.keys())

    return {"status": "all aggregators and candidate selection passed"}


# ==============================================================================
# J. SERVER ORCHESTRATION & CHECKPOINTING TESTS
# ==============================================================================
def test_server_orchestration():
    results = {}
    val_x = torch.randn(20, 1, 28, 28)
    val_y = torch.randint(0, 10, (20,))
    val_ds = TensorDataset(val_x, val_y)

    server = Server(
        model="mnist",
        detector_type="multimetric",
        trim_ratio=0.25,
        quarantine_threshold=0.20
    )

    # 1. Evaluation
    loss, acc = server.evaluate(val_ds, return_loss=True)
    assert isinstance(loss, float) and not math.isnan(loss)
    assert 0.0 <= acc <= 1.0

    # 2. Checkpoint Save & Restore
    server.trust_engine.history.update_trust(client_id=1, trust_score=0.91)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        chk_path = tmp.name
    try:
        server.save_checkpoint(chk_path, round_number=5, extra_meta={"dataset": "mnist"})
        
        fresh_server = Server(model="mnist", detector_type="multimetric")
        restored_round = fresh_server.load_checkpoint(chk_path)
        
        assert restored_round == 5
        assert fresh_server.trust_engine.history.get_latest_trust(1) == 0.91
        diff = torch.norm(server.global_model.get_parameter_vector() - fresh_server.global_model.get_parameter_vector()).item()
        assert diff < 1e-6, f"Server parameter restore diff {diff} >= 1e-6"
    finally:
        if os.path.exists(chk_path):
            os.remove(chk_path)

    results["eval_loss"] = loss
    results["eval_acc"] = acc
    results["restored_round"] = restored_round
    results["checkpoint_diff"] = diff
    return results


def main():
    print("RUNNING TARA-FL QA PHASE 3 FUNCTIONAL VALIDATION...")
    sections = [
        ("A_MODEL_LAYER", test_model_layer),
        ("B_CLIENT_LAYER", test_client_layer),
        ("C_DATA_PARTITIONING", test_data_partitioning),
        ("D_ATTACK_SUITE", test_attack_suite),
        ("E_DETECTION_ENGINE", test_detection_engine),
        ("F_TRUST_ENGINE_AND_QUARANTINE", test_trust_engine_and_quarantine),
        ("G_VALIDATION_PROBE", test_validation_probe),
        ("H_ROUND_RISK_AND_THREATS", test_round_risk_and_threats),
        ("I_AGGREGATION_SUITE", test_aggregation_suite),
        ("J_SERVER_ORCHESTRATION", test_server_orchestration)
    ]

    all_results = {}
    passed_count = 0
    failed_count = 0

    for name, fn in sections:
        res = run_test_section(name, fn)
        all_results[name] = res
        if res["status"] == "PASS":
            passed_count += 1
        else:
            failed_count += 1

    out_file = os.path.join(os.path.dirname(__file__), "qa_phase3_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\nPHASE 3 COMPLETE: {passed_count}/{len(sections)} PASSED, {failed_count} FAILED.\nSaved results to: {out_file}\n{'='*60}")


if __name__ == "__main__":
    main()
