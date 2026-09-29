import random
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"


# ==============================================================================
# DATASET LOADERS
# ==============================================================================

def load_mnist() -> Tuple[Dataset, Dataset]:
    """Load standard MNIST dataset."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset


def load_fashion_mnist() -> Tuple[Dataset, Dataset]:
    """Load Fashion-MNIST dataset with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_dataset = datasets.FashionMNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset


def load_cifar10() -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset with standard 3-channel normalization."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transform_test
    )
    return train_dataset, test_dataset


def load_dataset(name: str = "mnist") -> Tuple[Dataset, Dataset]:
    """
    Unified dataset loader supporting 'mnist', 'fashion_mnist', and 'cifar10'.
    """
    key = name.lower().strip()
    if key in ["mnist"]:
        return load_mnist()
    elif key in ["fashion_mnist", "fashionmnist", "fashion"]:
        return load_fashion_mnist()
    elif key in ["cifar10", "cifar"]:
        return load_cifar10()
    else:
        raise ValueError(
            f"Unsupported dataset '{name}'. Available: ['mnist', 'fashion_mnist', 'cifar10']"
        )


# ==============================================================================
# IID CLIENT PARTITIONING
# ==============================================================================

def create_clients(train_dataset: Dataset, num_clients: int = 10) -> List[Dataset]:
    """
    Split training data uniformly and independently across clients (IID).
    """
    total = len(train_dataset)
    base_size = total // num_clients
    remainder = total % num_clients

    sizes = [
        base_size + (1 if i < remainder else 0)
        for i in range(num_clients)
    ]
    return list(random_split(train_dataset, sizes))


# ==============================================================================
# NON-IID CLIENT PARTITIONING (DIRICHLET)
# ==============================================================================

def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Helper to extract integer label array from various PyTorch Dataset types."""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    elif hasattr(dataset, 'dataset'):
        full_labels = _extract_labels(dataset.dataset)
        if hasattr(dataset, 'indices'):
            return full_labels[dataset.indices]
        return full_labels
    return np.array([dataset[i][1] for i in range(len(dataset))])


def create_clients_noniid(
        train_dataset: Dataset,
        num_clients: int = 10,
        alpha: float = 0.5,
        seed: int = 42
) -> List[Dataset]:
    """
    Non-IID partitioning using Dirichlet distribution across class labels.
    
    alpha -> inf : Uniform IID
    alpha = 1.0  : Mild Non-IID
    alpha = 0.5  : Moderate Non-IID
    alpha = 0.1  : Severe Non-IID
    """
    rng = np.random.default_rng(seed)
    labels = _extract_labels(train_dataset)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        counts = (proportions * len(class_indices)).astype(int)
        counts[-1] = len(class_indices) - counts[:-1].sum()

        current = 0
        for client_id in range(num_clients):
            num_samples = counts[client_id]
            if num_samples > 0:
                client_indices[client_id].extend(
                    class_indices[current:current + num_samples].tolist()
                )
            current += num_samples

    client_datasets = []
    for indices in client_indices:
        rng.shuffle(indices)
        client_datasets.append(Subset(train_dataset, indices))

    return client_datasets


# ==============================================================================
# PATHOLOGICAL NON-IID PARTITIONING (Fixed Classes Per Client)
# ==============================================================================

def create_clients_pathological(
        train_dataset: Dataset,
        num_clients: int = 10,
        classes_per_client: int = 2,
        seed: int = 42
) -> List[Dataset]:
    """
    Pathological Non-IID partitioning where each client strictly receives samples
    from a small fixed number of classes (e.g. 2 classes per client out of 10).
    """
    rng = np.random.default_rng(seed)
    labels = _extract_labels(train_dataset)
    num_classes = len(np.unique(labels))

    # Sort indices by class label
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Assign classes to each client
    client_indices = [[] for _ in range(num_clients)]
    times_assigned = {c: 0 for c in range(num_classes)}
    total_shards = num_clients * classes_per_client
    shards_per_class = max(1, total_shards // num_classes)

    class_pool = []
    for c in range(num_classes):
        class_pool.extend([c] * shards_per_class)
    while len(class_pool) < total_shards:
        class_pool.append(len(class_pool) % num_classes)
    rng.shuffle(class_pool)

    # Shard sizes per class
    shards = {c: [] for c in range(num_classes)}
    for c in range(num_classes):
        indices = class_indices[c]
        count = class_pool.count(c)
        if count > 0:
            shard_size = max(1, len(indices) // count)
            for i in range(count):
                start = i * shard_size
                end = (i + 1) * shard_size if i < count - 1 else len(indices)
                shards[c].append(indices[start:end])

    shard_tracker = {c: 0 for c in range(num_classes)}
    for i in range(num_clients):
        assigned_classes = class_pool[i * classes_per_client:(i + 1) * classes_per_client]
        for c in assigned_classes:
            idx = shard_tracker[c]
            if idx < len(shards[c]):
                client_indices[i].extend(shards[c][idx])
                shard_tracker[c] += 1

    return [Subset(train_dataset, idxs) for idxs in client_indices]


# ==============================================================================
# CLIENT SAMPLING & NETWORK SIMULATION HELPERS
# ==============================================================================

def sample_active_clients(
        client_ids: List[int],
        sample_ratio: float = 1.0,
        seed: Optional[int] = None
) -> List[int]:
    """
    Subsample a fraction C of clients for the current communication round.
    """
    if sample_ratio >= 1.0:
        return list(client_ids)

    k = max(1, int(len(client_ids) * sample_ratio))
    rng = random.Random(seed)
    return sorted(rng.sample(client_ids, k))


def simulate_client_dropout(
        client_ids: List[int],
        dropout_rate: float = 0.1,
        seed: Optional[int] = None
) -> List[int]:
    """
    Simulate unreliable edge devices / network dropouts where a fraction
    of clients fail to upload updates in the current round.
    """
    if dropout_rate <= 0.0:
        return list(client_ids)

    rng = random.Random(seed)
    surviving = [c_id for c_id in client_ids if rng.random() > dropout_rate]
    if not surviving and client_ids:
        # Guarantee at least 1 client communicates
        surviving = [rng.choice(client_ids)]
    return surviving


# Aliases for functional API consistency
partition_iid = create_clients
partition_dirichlet = create_clients_noniid
partition_pathological = create_clients_pathological
simulate_dropout = simulate_client_dropout

