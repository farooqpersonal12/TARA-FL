import torch
import numpy as np
#-->Main ML framework for our project implementation

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

from torchvision import datasets, transforms

""""
->provides computer vision data sets which is MNIST and 
transforms image dataset to python tensor 
"""


def load_mnist():  #-> loads MNIST dataset from torchvision
    transform = transforms.ToTensor()

    # dividing dataset into training and testing set
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


# ==============================================================
# IID CLIENT PARTITIONING
# ==============================================================

def create_clients(train_dataset, num_clients=10):
    """
    Split training data uniformly across clients (IID).

    Handles the case where the dataset size is not evenly
    divisible by num_clients.
    """

    total = len(train_dataset)
    base_size = total // num_clients
    remainder = total % num_clients

    # First 'remainder' clients get one extra sample.
    sizes = [
        base_size + (1 if i < remainder else 0)
        for i in range(num_clients)
    ]

    client_datasets = torch.utils.data.random_split(
        train_dataset,
        sizes
    )

    return client_datasets


# ==============================================================
# NON-IID CLIENT PARTITIONING (DIRICHLET)
# ==============================================================

def create_clients_noniid(
        train_dataset,
        num_clients=10,
        alpha=0.5,
        seed=42
):
    """
    Non-IID partitioning using Dirichlet distribution.

    Lower alpha = more heterogeneous.

        alpha = 100.0  → nearly IID
        alpha = 1.0    → mild heterogeneity
        alpha = 0.5    → moderate heterogeneity
        alpha = 0.1    → severe heterogeneity

    Each client receives a different proportion of each
    digit class, controlled by the Dirichlet parameter.
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------
    # Extract all labels
    # ----------------------------------------------------------

    if hasattr(train_dataset, 'targets'):

        labels = np.array(train_dataset.targets)

    elif hasattr(train_dataset, 'dataset'):

        labels = np.array(
            train_dataset.dataset.targets
        )

    else:

        labels = np.array([
            train_dataset[i][1]
            for i in range(len(train_dataset))
        ])

    num_classes = len(np.unique(labels))

    # ----------------------------------------------------------
    # Dirichlet allocation
    # ----------------------------------------------------------

    client_indices = [[] for _ in range(num_clients)]

    for class_label in range(num_classes):

        class_indices = np.where(
            labels == class_label
        )[0]

        rng.shuffle(class_indices)

        # Sample proportions from Dirichlet distribution.
        proportions = rng.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # Convert proportions to actual sample counts.
        proportions = (
                proportions
                / proportions.sum()
        )

        # Calculate the number of samples per client
        # for this class.
        counts = (
                proportions * len(class_indices)
        ).astype(int)

        # Assign any remainder to the last client.
        counts[-1] = (
                len(class_indices) - counts[:-1].sum()
        )

        # Distribute indices.
        current = 0

        for client_id in range(num_clients):

            num_samples = counts[client_id]

            client_indices[client_id].extend(
                class_indices[
                    current:current + num_samples
                ].tolist()
            )

            current += num_samples

    # ----------------------------------------------------------
    # Create Subset datasets
    # ----------------------------------------------------------

    client_datasets = []

    for indices in client_indices:

        rng.shuffle(indices)

        subset = torch.utils.data.Subset(
            train_dataset,
            indices
        )

        client_datasets.append(subset)

    return client_datasets
