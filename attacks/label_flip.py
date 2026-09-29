import random
from typing import Optional, Dict
import torch
from torch.utils.data import Dataset
from attacks.base_attack import BaseAttack


class LabelFlipDataset(Dataset):
    """
    Data poisoning dataset wrapper that flips labels of a specified fraction of samples.
    
    Supports:
      - Systematic shift: label -> (label + shift) % num_classes
      - Target mapping: e.g. {1: 7, 2: 8} to swap specific classes
    """

    def __init__(
            self,
            dataset: Dataset,
            flip_ratio: float = 0.5,
            num_classes: int = 10,
            shift: int = 1,
            target_map: Optional[Dict[int, int]] = None,
            seed: int = 42
    ):
        self.dataset = dataset
        self.flip_ratio = flip_ratio
        self.num_classes = num_classes
        self.shift = shift
        self.target_map = target_map

        num_samples = len(dataset)
        num_flips = int(num_samples * flip_ratio)

        random_generator = random.Random(seed)
        self.flip_indices = set(
            random_generator.sample(range(num_samples), num_flips)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]

        if index in self.flip_indices:
            is_tensor = isinstance(label, torch.Tensor)
            raw_label = int(label.item()) if is_tensor else int(label)

            if self.target_map and raw_label in self.target_map:
                new_label = self.target_map[raw_label]
            else:
                new_label = (raw_label + self.shift) % self.num_classes

            if is_tensor:
                label = torch.tensor(new_label, dtype=label.dtype)
            else:
                label = new_label

        return image, label


class LabelFlipAttack(BaseAttack):
    """
    Label Flip attack orchestrator.
    """

    def __init__(
            self,
            flip_ratio: float = 0.5,
            num_classes: int = 10,
            shift: int = 1,
            target_map: Optional[Dict[int, int]] = None,
            seed: int = 42
    ):
        super().__init__(name="LabelFlipAttack")
        self.flip_ratio = flip_ratio
        self.num_classes = num_classes
        self.shift = shift
        self.target_map = target_map
        self.seed = seed

    def apply(self, dataset: Dataset) -> Dataset:
        return LabelFlipDataset(
            dataset=dataset,
            flip_ratio=self.flip_ratio,
            num_classes=self.num_classes,
            shift=self.shift,
            target_map=self.target_map,
            seed=self.seed
        )