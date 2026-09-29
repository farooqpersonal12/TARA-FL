from typing import Dict, Type, List
from aggregation.base_aggregator import BaseAggregator
from aggregation.strategies import (
    TrustAwareFedAvg,
    TrustWeightedRobustTrimming,
    TrustWeightedMedian,
    TrustAwareFedAvgM,
    MultiKrumAggregator
)


class AggregatorFactory:
    """Factory for instantiating aggregation algorithms."""

    _registry: Dict[str, Type[BaseAggregator]] = {
        "trust_aware_fedavg": TrustAwareFedAvg,
        "fedavg": TrustAwareFedAvg,
        "trust_weighted_robust": TrustWeightedRobustTrimming,
        "robust": TrustWeightedRobustTrimming,
        "robust_trim": TrustWeightedRobustTrimming,
        "trust_weighted_median": TrustWeightedMedian,
        "median": TrustWeightedMedian,
        "fedavgm": TrustAwareFedAvgM,
        "momentum": TrustAwareFedAvgM,
        "multi_krum": MultiKrumAggregator,
        "krum": MultiKrumAggregator,
    }

    @classmethod
    def register(cls, name: str, aggregator_class: Type[BaseAggregator]):
        cls._registry[name.lower().strip()] = aggregator_class

    @classmethod
    def list_aggregators(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAggregator:
        key = name.lower().strip()
        if key not in cls._registry:
            available = ", ".join(cls.list_aggregators())
            raise ValueError(f"Unknown aggregator '{name}'. Available: [{available}]")
        return cls._registry[key](**kwargs)


def get_aggregator(name: str = "trust_aware_fedavg", **kwargs) -> BaseAggregator:
    return AggregatorFactory.create(name, **kwargs)
