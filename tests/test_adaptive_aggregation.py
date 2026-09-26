import torch

from aggregation.adaptive_aggregator import AdaptiveAggregator


# ==========================================================
# HELPER
# ==========================================================

def create_parameters(values):

    return {
        "weight": torch.tensor(values)
    }


# ==========================================================
# AGGREGATOR SELECTION TESTS
# ==========================================================

def test_low_risk_selects_trust_aware_fedavg():

    aggregator = AdaptiveAggregator()

    assert (
            aggregator.select_aggregator("LOW")
            == "TRUST_AWARE_FEDAVG"
    )


def test_medium_risk_selects_trust_weighted_robust():

    aggregator = AdaptiveAggregator()

    assert (
            aggregator.select_aggregator("MEDIUM")
            == "TRUST_WEIGHTED_ROBUST"
    )


def test_high_risk_selects_trust_weighted_median():

    aggregator = AdaptiveAggregator()

    assert (
            aggregator.select_aggregator("HIGH")
            == "TRUST_WEIGHTED_MEDIAN"
    )


# ==========================================================
# TRUST WEIGHT TEST
# ==========================================================

def test_calculate_trust_weights():

    aggregator = AdaptiveAggregator()

    client_sizes = [
        100,
        100,
        100
    ]

    trust_scores = {
        1: 1.0,
        2: 0.5,
        3: 0.0
    }

    weights = aggregator.calculate_trust_weights(
        client_sizes,
        trust_scores
    )

    assert len(weights) == 3

    assert abs(
        sum(weights) - 1.0
    ) < 1e-6

    assert weights[0] > weights[1]
    assert weights[1] > weights[2]


# ==========================================================
# TRUST-AWARE FEDAVG TEST
# ==========================================================

def test_trust_aware_fedavg():

    aggregator = AdaptiveAggregator()

    clients = [
        create_parameters([1.0, 1.0]),
        create_parameters([2.0, 2.0]),
        create_parameters([10.0, 10.0])
    ]

    client_sizes = [
        100,
        100,
        100
    ]

    trust_scores = {
        1: 1.0,
        2: 1.0,
        3: 0.0
    }

    result = aggregator.trust_aware_fedavg(
        clients,
        client_sizes,
        trust_scores
    )

    # Client 3 has zero trust and therefore
    # should not influence the aggregation.

    expected = torch.tensor([
        1.5,
        1.5
    ])

    assert torch.allclose(
        result["weight"],
        expected,
        atol=1e-6
    )


# ==========================================================
# UPDATE DISTANCE TEST
# ==========================================================

def test_calculate_update_distances():

    aggregator = AdaptiveAggregator()

    client_updates = {
        1: create_parameters([1.0, 1.0]),
        2: create_parameters([1.0, 1.0]),
        3: create_parameters([10.0, 10.0])
    }

    distances = (
        aggregator.calculate_update_distances(
            client_updates
        )
    )

    assert len(distances) == 3

    assert distances[3] > distances[1]
    assert distances[3] > distances[2]


# ==========================================================
# ROBUST CLIENT SELECTION TEST
# ==========================================================

def test_select_robust_clients():

    aggregator = AdaptiveAggregator(
        trim_ratio=0.25
    )

    client_ids = [
        1,
        2,
        3,
        4
    ]

    distances = {
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 10.0
    }

    selected = aggregator.select_robust_clients(
        client_ids,
        distances,
        "MEDIUM"
    )

    assert 4 not in selected

    assert len(selected) == 3


# ==========================================================
# TRUST-WEIGHTED ROBUST TEST
# ==========================================================

def test_trust_weighted_robust():

    aggregator = AdaptiveAggregator()

    client_parameters = [
        create_parameters([1.0, 1.0]),
        create_parameters([2.0, 2.0]),
        create_parameters([3.0, 3.0]),
        create_parameters([10.0, 10.0])
    ]

    client_updates = {
        1: create_parameters([1.0, 1.0]),
        2: create_parameters([2.0, 2.0]),
        3: create_parameters([3.0, 3.0]),
        4: create_parameters([10.0, 10.0])
    }

    client_sizes = [
        100,
        100,
        100,
        100
    ]

    trust_scores = {
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 0.0
    }

    (
        result,
        selected_clients,
        distances
    ) = aggregator.trust_weighted_robust(
        client_parameters,
        client_sizes,
        trust_scores,
        client_updates,
        "MEDIUM"
    )

    assert result is not None

    assert 4 not in selected_clients

    assert len(selected_clients) == 3

    assert distances[4] > distances[1]

# ==========================================================
# TRUST-WEIGHTED MEDIAN TEST
# ==========================================================

def test_trust_weighted_median():

    aggregator = AdaptiveAggregator()

    clients = [
        create_parameters([1.0, 10.0]),
        create_parameters([2.0, 20.0]),
        create_parameters([100.0, 200.0])
    ]

    client_sizes = [
        100,
        100,
        100
    ]

    trust_scores = {
        1: 1.0,
        2: 1.0,
        3: 0.0
    }

    result = aggregator.trust_weighted_median(
        clients,
        client_sizes,
        trust_scores
    )

    expected = torch.tensor([
        1.0,
        10.0
    ])

    assert torch.allclose(
        result["weight"],
        expected,
        atol=1e-6
    )


# ==========================================================
# COMPLETE ADAPTIVE AGGREGATION TEST
# ==========================================================

def test_complete_adaptive_aggregation():

    aggregator = AdaptiveAggregator()

    client_parameters = [
        create_parameters([1.0, 1.0]),
        create_parameters([2.0, 2.0]),
        create_parameters([10.0, 10.0])
    ]

    client_updates = {
        1: create_parameters([1.0, 1.0]),
        2: create_parameters([2.0, 2.0]),
        3: create_parameters([10.0, 10.0])
    }

    client_sizes = [
        100,
        100,
        100
    ]

    trust_scores = {
        1: 1.0,
        2: 1.0,
        3: 0.0
    }

    for risk_level in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]:

        result, aggregator_name = (
            aggregator.aggregate(
                client_parameters,
                client_sizes,
                trust_scores,
                risk_level,
                client_updates
            )
        )

        assert result is not None

        assert aggregator_name in [
            "TRUST_AWARE_FEDAVG",
            "TRUST_WEIGHTED_ROBUST",
            "TRUST_WEIGHTED_MEDIAN"
        ]