"""
TARA-FL Aggregation Package

Provides BaseAggregator, AdaptiveAggregator, individual strategies (TrustAwareFedAvg,
TrustWeightedRobustTrimming, TrustWeightedMedian, TrustAwareFedAvgM, MultiKrumAggregator),
CandidateModelSelector, and AggregatorFactory.
"""

from aggregation.base_aggregator import BaseAggregator
from aggregation.adaptive_aggregator import AdaptiveAggregator
from aggregation.strategies import (
    TrustAwareFedAvg,
    TrustWeightedRobustTrimming,
    TrustWeightedMedian,
    TrustAwareFedAvgM,
    MultiKrumAggregator
)
from aggregation.candidate_selector import CandidateModelSelector
from aggregation.aggregator_factory import AggregatorFactory, get_aggregator

__all__ = [
    "BaseAggregator",
    "AdaptiveAggregator",
    "TrustAwareFedAvg",
    "TrustWeightedRobustTrimming",
    "TrustWeightedMedian",
    "TrustAwareFedAvgM",
    "MultiKrumAggregator",
    "CandidateModelSelector",
    "AggregatorFactory",
    "get_aggregator"
]
