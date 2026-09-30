import pytest
import torch
from torch.utils.data import TensorDataset

from model.model import MNISTModel
from risk.round_risk import RoundRisk
from risk.threat_classifier import ThreatClassifier, ThreatType
from aggregation.strategies import (
    TrustAwareFedAvg,
    TrustWeightedRobustTrimming,
    TrustWeightedMedian,
    TrustAwareFedAvgM,
    MultiKrumAggregator
)
from aggregation.adaptive_aggregator import AdaptiveAggregator
from aggregation.candidate_selector import CandidateModelSelector
from aggregation.aggregator_factory import AggregatorFactory, get_aggregator


@pytest.fixture
def mock_params_and_trust():
    model = MNISTModel()
    base_state = model.state_dict()

    p1 = {k: v.clone() + 0.01 for k, v in base_state.items()}
    p2 = {k: v.clone() - 0.01 for k, v in base_state.items()}
    p3 = {k: v.clone() + 0.02 for k, v in base_state.items()}
    p4 = {k: v.clone() + 5.0 for k, v in base_state.items()}  # Adversary

    client_params = [p1, p2, p3, p4]
    client_sizes = [100, 100, 100, 100]
    trust_scores = {1: 0.95, 2: 0.92, 3: 0.88, 4: 0.10}

    u1 = {k: p1[k] - base_state[k] for k in p1}
    u2 = {k: p2[k] - base_state[k] for k in p2}
    u3 = {k: p3[k] - base_state[k] for k in p3}
    u4 = {k: p4[k] - base_state[k] for k in p4}
    client_updates = {1: u1, 2: u2, 3: u3, 4: u4}

    return client_params, client_sizes, trust_scores, client_updates, base_state


class TestRoundRiskAndThreats:
    """Test suite for Round-Risk Engine and Threat Classifier."""

    def test_round_risk_regimes(self):
        risk_engine = RoundRisk()

        # Low risk conditions
        dist_low = {1: 0.1, 2: 0.12, 3: 0.09}
        trust_high = {1: 0.95, 2: 0.92, 3: 0.90}
        score, level, susp = risk_engine.calculate_risk(dist_low, trust_high)
        assert level == "LOW"
        assert susp == 0

        # High risk conditions (adversary present)
        dist_high = {1: 0.1, 2: 0.1, 3: 0.1, 4: 10.0}
        trust_low = {1: 0.95, 2: 0.92, 3: 0.90, 4: 0.05}
        score, level, susp = risk_engine.calculate_risk(dist_high, trust_low)
        assert level in ["MEDIUM", "HIGH"]
        assert susp == 1

    def test_low_risk_round(self):
        risk_engine = RoundRisk(low_risk_threshold=0.30, medium_risk_threshold=0.60)
        # 5 honest clients with small peer distances and near-perfect trust
        distances = {1: 0.05, 2: 0.04, 3: 0.06, 4: 0.05, 5: 0.05}
        trust_scores = {1: 0.98, 2: 0.97, 3: 0.95, 4: 0.96, 5: 0.98}

        score, level, susp = risk_engine.calculate_risk(distances, trust_scores)
        assert 0.0 <= score < 0.30
        assert level == "LOW"
        assert susp == 0

        # Verify aggregator selection
        aggregator = AdaptiveAggregator()
        selected = aggregator.select_aggregator(level)
        assert selected == "TRUST_AWARE_FEDAVG"

    def test_medium_risk_round(self):
        risk_engine = RoundRisk(low_risk_threshold=0.30, medium_risk_threshold=0.60)
        # Moderate distances and mild trust degradation (1 suspicious client)
        distances = {1: 0.8, 2: 0.9, 3: 0.85, 4: 2.5}
        trust_scores = {1: 0.80, 2: 0.78, 3: 0.82, 4: 0.45}

        score, level, susp = risk_engine.calculate_risk(distances, trust_scores)
        assert 0.30 <= score < 0.60
        assert level == "MEDIUM"
        assert susp == 1

        # Verify aggregator selection
        aggregator = AdaptiveAggregator()
        selected = aggregator.select_aggregator(level)
        assert selected == "TRUST_WEIGHTED_ROBUST"

    def test_high_risk_round(self):
        risk_engine = RoundRisk(low_risk_threshold=0.30, medium_risk_threshold=0.60)
        # Severe divergence with multiple compromised clients and large anomaly distances
        distances = {1: 5.0, 2: 6.0, 3: 7.5, 4: 8.0, 5: 0.1}
        trust_scores = {1: 0.10, 2: 0.08, 3: 0.05, 4: 0.12, 5: 0.90}

        score, level, susp = risk_engine.calculate_risk(distances, trust_scores)
        assert score >= 0.60
        assert level == "HIGH"
        assert susp == 4

        # Verify aggregator selection
        aggregator = AdaptiveAggregator()
        selected = aggregator.select_aggregator(level)
        assert selected == "TRUST_WEIGHTED_MEDIAN"

    def test_missing_optional_metrics(self):
        risk_engine = RoundRisk()

        # 1. Both distances and trust_scores missing
        score, level, susp = risk_engine.calculate_risk(distances=None, trust_scores=None)
        assert score == 0.0
        assert level == "LOW"
        assert susp == 0

        # 2. Distances missing (None or empty dict), trust_scores provided
        trust_scores = {1: 0.95, 2: 0.90, 3: 0.20}
        score, level, susp = risk_engine.calculate_risk(distances=None, trust_scores=trust_scores)
        assert 0.0 <= score <= 1.0
        assert level in ["LOW", "MEDIUM", "HIGH"]
        assert susp == 1

        # 3. Trust scores missing, distances provided
        distances = {1: 0.1, 2: 5.0}
        score, level, susp = risk_engine.calculate_risk(distances=distances, trust_scores=None)
        assert 0.0 <= score <= 1.0

        # 4. Single client metric
        score_single, level_single, _ = risk_engine.calculate_risk(
            distances={1: 0.0},
            trust_scores={1: 1.0},
            current_accuracy=None,
            client_updates=None
        )
        assert score_single == 0.0
        assert level_single == "LOW"

        # 5. Invalid accuracy value handled safely
        score_acc, _, _ = risk_engine.calculate_risk(
            distances={1: 0.2},
            trust_scores={1: 0.9},
            current_accuracy=-0.5
        )
        assert 0.0 <= score_acc <= 1.0

    def test_aggregator_selection_from_calculated_risk(self, mock_params_and_trust):
        client_params, client_sizes, _, client_updates, _ = mock_params_and_trust
        aggregator = AdaptiveAggregator()

        # Low risk scenario -> TrustAwareFedAvg
        risk_engine = RoundRisk()
        _, level_low, _ = risk_engine.calculate_risk(
            distances={1: 0.05, 2: 0.05, 3: 0.05},
            trust_scores={1: 0.95, 2: 0.95, 3: 0.95}
        )
        assert level_low == "LOW"
        _, selected_low = aggregator.aggregate(
            client_parameters=client_params[:3],
            client_sizes=client_sizes[:3],
            trust_scores={1: 0.95, 2: 0.95, 3: 0.95},
            risk_level=level_low
        )
        assert selected_low == "TRUST_AWARE_FEDAVG"

        # Medium risk scenario -> TrustWeightedRobust
        _, level_med, _ = risk_engine.calculate_risk(
            distances={1: 0.5, 2: 0.8, 3: 2.0},
            trust_scores={1: 0.85, 2: 0.80, 3: 0.45}
        )
        assert level_med == "MEDIUM"
        _, selected_med = aggregator.aggregate(
            client_parameters=client_params[:3],
            client_sizes=client_sizes[:3],
            trust_scores={1: 0.85, 2: 0.80, 3: 0.45},
            risk_level=level_med,
            client_updates={1: client_updates[1], 2: client_updates[2], 3: client_updates[3]}
        )
        assert selected_med == "TRUST_WEIGHTED_ROBUST"

        # High risk scenario -> TrustWeightedMedian
        _, level_high, _ = risk_engine.calculate_risk(
            distances={1: 6.0, 2: 7.0, 3: 8.0, 4: 0.1},
            trust_scores={1: 0.1, 2: 0.1, 3: 0.05, 4: 0.95}
        )
        assert level_high == "HIGH"
        _, selected_high = aggregator.aggregate(
            client_parameters=client_params,
            client_sizes=client_sizes,
            trust_scores={1: 0.1, 2: 0.1, 3: 0.05, 4: 0.95},
            risk_level=level_high
        )
        assert selected_high == "TRUST_WEIGHTED_MEDIAN"

    def test_round_risk_details_breakdown(self):
        risk_engine = RoundRisk()
        dist = {1: 0.5, 2: 1.5}
        trust = {1: 0.9, 2: 0.3}

        score, level, susp = risk_engine.calculate_risk(
            distances=dist,
            trust_scores=trust,
            current_accuracy=0.90
        )
        details = risk_engine.get_last_details()

        assert "risk_score" in details
        assert "risk_level" in details
        assert "threat_type" in details
        assert "anomaly_component" in details
        assert "trust_component" in details
        assert "suspicious_ratio" in details
        assert "worst_client_component" in details
        assert details["risk_score"] == score
        assert details["risk_level"] == level

    def test_performance_feedback_risk_escalation(self):
        risk_engine = RoundRisk(enable_performance_feedback=True)
        dist = {1: 0.5, 2: 0.5}
        trust = {1: 0.70, 2: 0.70}

        # Round 1: 90% accuracy
        score1, _, _ = risk_engine.calculate_risk(dist, trust, current_accuracy=0.90)

        # Round 2: Drops to 70% accuracy (22% drop -> should escalate risk)
        score2, _, _ = risk_engine.calculate_risk(dist, trust, current_accuracy=0.70)
        assert score2 > score1

    def test_threat_classifier(self):
        classifier = ThreatClassifier()

        # Clean
        t_clean = classifier.classify_threat({1: 0.2, 2: 0.2}, {1: 0.95, 2: 0.95})
        assert t_clean == ThreatType.CLEAN

        # Magnitude scaling
        t_scale = classifier.classify_threat({1: 0.2, 2: 8.5}, {1: 0.95, 2: 0.10})
        assert t_scale == ThreatType.MAGNITUDE_SCALING


class TestAggregationStrategiesSuite:
    """Test suite for all individual and adaptive aggregators."""

    def test_trust_aware_fedavg(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, _, _ = mock_params_and_trust
        agg = TrustAwareFedAvg()
        res = agg.aggregate(client_params, client_sizes, trust_scores)
        assert len(res) == len(client_params[0])

    def test_trust_weighted_robust(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, client_updates, _ = mock_params_and_trust
        agg = TrustWeightedRobustTrimming(trim_ratio=0.25)
        res = agg.aggregate(client_params, client_sizes, trust_scores, client_updates=client_updates)
        assert len(res) == len(client_params[0])

    def test_trust_weighted_median(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, _, _ = mock_params_and_trust
        agg = TrustWeightedMedian()
        res = agg.aggregate(client_params, client_sizes, trust_scores)
        assert len(res) == len(client_params[0])

    def test_trust_aware_fedavgm_momentum(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, _, base_state = mock_params_and_trust
        agg = TrustAwareFedAvgM(beta=0.9, server_lr=1.0)

        # Round 1
        res1 = agg.aggregate(client_params, client_sizes, trust_scores, global_parameters=base_state)
        # Round 2
        res2 = agg.aggregate(client_params, client_sizes, trust_scores, global_parameters=res1)
        assert len(res2) == len(base_state)

    def test_multi_krum(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, _, _ = mock_params_and_trust
        agg = MultiKrumAggregator(num_malicious=1)
        res = agg.aggregate(client_params, client_sizes, trust_scores)
        assert len(res) == len(client_params[0])

    def test_adaptive_aggregator_routing(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, client_updates, _ = mock_params_and_trust
        adaptive = AdaptiveAggregator()

        # LOW -> FedAvg
        _, strat_low = adaptive.aggregate(client_params, client_sizes, trust_scores, risk_level="LOW")
        assert strat_low == "TRUST_AWARE_FEDAVG"

        # MEDIUM -> Robust Trimming
        _, strat_med = adaptive.aggregate(
            client_params, client_sizes, trust_scores, risk_level="MEDIUM", client_updates=client_updates
        )
        assert strat_med == "TRUST_WEIGHTED_ROBUST"

        # HIGH -> Weighted Median
        _, strat_high = adaptive.aggregate(client_params, client_sizes, trust_scores, risk_level="HIGH")
        assert strat_high == "TRUST_WEIGHTED_MEDIAN"

    def test_candidate_selector_multi_model(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, client_updates, _ = mock_params_and_trust
        dummy_probe = TensorDataset(torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,)))

        selector = CandidateModelSelector(dummy_probe)
        model_template = MNISTModel()

        best_params, best_name, scores = selector.select_best_candidate(
            model_template=model_template,
            client_parameters=client_params,
            client_sizes=client_sizes,
            trust_scores=trust_scores,
            client_updates=client_updates
        )

        assert best_name in selector.candidates
        assert len(scores) == 4
        assert len(best_params) == len(model_template.state_dict())

    def test_aggregator_factory(self):
        aggs = AggregatorFactory.list_aggregators()
        assert "fedavg" in aggs
        assert "robust" in aggs
        assert "median" in aggs
        assert "multi_krum" in aggs

        agg_obj = get_aggregator("robust")
        assert isinstance(agg_obj, TrustWeightedRobustTrimming)

    def test_adaptive_aggregator_with_quarantine_manager(self, mock_params_and_trust):
        client_params, client_sizes, trust_scores, client_updates, base_state = mock_params_and_trust
        from trustengine.quarantine import QuarantineManager

        qm = QuarantineManager(quarantine_threshold=0.20)
        qm.evaluate_client(1, 0.95, 1.0)
        qm.evaluate_client(2, 0.92, 1.0)
        qm.evaluate_client(3, 0.88, 1.0)
        qm.evaluate_client(4, 0.05, 10.0)  # Quarantined

        adaptive = AdaptiveAggregator()
        res, strat = adaptive.aggregate(
            client_parameters=client_params,
            client_sizes=client_sizes,
            trust_scores=trust_scores,
            risk_level="LOW",
            client_updates=client_updates,
            quarantine_manager=qm
        )
        assert strat == "TRUST_AWARE_FEDAVG"
        # Verify client 4 (+5.0) was excluded
        for k in res:
            diff = (res[k] - base_state[k]).abs().max().item()
            assert diff < 0.1, f"Quarantined client 4 leaked into aggregation: diff={diff}"

    def test_non_contiguous_client_ids_aggregation(self):
        """Test aggregators with non-contiguous client IDs (e.g. [2, 5, 9])."""
        p2 = {"w": torch.tensor([1.0, 1.0])}
        p5 = {"w": torch.tensor([2.0, 2.0])}
        p9 = {"w": torch.tensor([3.0, 3.0])}

        params = [p2, p5, p9]
        sizes = [100, 100, 100]
        trust = {2: 0.90, 5: 0.90, 9: 0.90}
        cids = [2, 5, 9]

        robust = TrustWeightedRobustTrimming()
        res_rob = robust.aggregate(params, sizes, trust, client_ids=cids)
        assert torch.allclose(res_rob["w"], torch.tensor([2.0, 2.0]))

        krum = MultiKrumAggregator(num_malicious=0, to_select=2)
        res_krum = krum.aggregate(params, sizes, trust, client_ids=cids)
        assert res_krum["w"] is not None
