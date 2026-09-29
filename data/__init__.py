"""
TARA-FL Data Package

Provides dataset loaders (MNIST, Fashion-MNIST, CIFAR-10) and partitioning strategies
(IID, Dirichlet Non-IID, Pathological Non-IID).
"""

from data.dataset import (
    load_mnist,
    load_fashion_mnist,
    load_cifar10,
    load_dataset,
    create_clients,
    create_clients_noniid,
    create_clients_pathological,
    sample_active_clients,
    simulate_client_dropout,
    partition_iid,
    partition_dirichlet,
    partition_pathological,
    simulate_dropout
)

__all__ = [
    "load_mnist",
    "load_fashion_mnist",
    "load_cifar10",
    "load_dataset",
    "create_clients",
    "create_clients_noniid",
    "create_clients_pathological",
    "sample_active_clients",
    "simulate_client_dropout",
    "partition_iid",
    "partition_dirichlet",
    "partition_pathological",
    "simulate_dropout"
]
