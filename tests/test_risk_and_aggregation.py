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
