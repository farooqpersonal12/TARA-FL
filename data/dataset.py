import torch
#-->Main ML framework for our project implementation

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

from torchvision import datasets, transforms

""""
->provides computer vision data sets which is MNIST and 
transforms image dataset to python tensor 
"""


def load_mnist():  #-> loads MNIST dataset form torchvision
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


#dividing training dataset for each client
def create_clients(train_dataset, num_clients=10):
    client_datasets = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) // num_clients] * num_clients
    )

    return client_datasets
