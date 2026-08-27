import random

import torch
from torch.utils.data import Dataset


class LabelFlipDataset(Dataset):

    def __init__(
            self,
            dataset,
            flip_ratio=0.5,
            seed=42
    ):
        self.dataset = dataset
        self.flip_ratio = flip_ratio

        num_samples = len(dataset)
        num_flips = int(num_samples * flip_ratio)

        random_generator = random.Random(seed)

        self.flip_indices = set(
            random_generator.sample(
                range(num_samples),
                num_flips
            )
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        if index in self.flip_indices:

            label = (label + 1) % 10

        return image, label