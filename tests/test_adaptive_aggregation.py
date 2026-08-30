import torch

from aggregation.adaptive_aggregator import (
    AdaptiveAggregator
)


def create_parameters(values):

    return {
        "weight": torch.tensor(
            values,
            dtype=torch.float32
        )
    }


def test_low_risk_selects_trust_aware_fedavg():

    aggregator = AdaptiveAggregator()

    assert (
            aggregator.select_aggregator("LOW")
            == "TRUST_AWARE_FEDAVG"
    )


def test_medium_risk_selects_trimmed_mean():

    aggregator = AdaptiveAggregator()

    assert (
            aggregator.select_aggregator("MEDIUM")
            == "TRIMMED_MEAN"
    )


def test_high_risk_selects_median():

    aggregator = AdaptiveAggregator()

    assert (
            aggregator.select_aggregator("HIGH")
            == "MEDIAN"
    )


def test_median_aggregation():

    aggregator = AdaptiveAggregator()

    clients = [
        create_parameters([1.0, 10.0]),
        create_parameters([2.0, 20.0]),
        create_parameters([3.0, 30.0])
    ]

    result = aggregator.median(clients)

    expected = torch.tensor(
        [2.0, 20.0]
    )

    assert torch.equal(
        result["weight"],
        expected
    )


def test_trust_aware_fedavg():

    aggregator = AdaptiveAggregator()

    clients = [
        create_parameters([1.0]),
        create_parameters([3.0])
    ]

    sizes = [100, 100]

    trust_scores = {
        1: 1.0,
        2: 0.0
    }

    result = aggregator.trust_aware_fedavg(
        clients,
        sizes,
        trust_scores
    )

    assert torch.allclose(
        result["weight"],
        torch.tensor([1.0])
    )