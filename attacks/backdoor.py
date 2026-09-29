import random
import torch
from torch.utils.data import Dataset
from attacks.base_attack import BaseAttack


class BackdoorDataset(Dataset):
    """
    Data poisoning dataset wrapper that injects a watermark trigger pattern
    into a fraction of images and re-labels them to a target class.
    
    Default trigger: A small white pixel patch (3x3) placed at the bottom-right corner.
    """

    def __init__(
            self,
            dataset: Dataset,
            poison_ratio: float = 0.5,
            target_label: int = 0,
            trigger_size: int = 3,
            trigger_value: float = 1.0,
            seed: int = 42
    ):
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

        num_samples = len(dataset)
        num_poisoned = int(num_samples * poison_ratio)

        random_gen = random.Random(seed)
        self.poison_indices = set(
            random_gen.sample(range(num_samples), num_poisoned)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _inject_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Inject square trigger in bottom-right corner."""
        img = image.clone()
        c, h, w = img.shape
        ts = self.trigger_size
        img[:, h - ts:h, w - ts:w] = self.trigger_value
        return img

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if index in self.poison_indices:
            image = self._inject_trigger(image)
            label = self.target_label
        return image, label


class BackdoorAttack(BaseAttack):
    """Backdoor attack orchestrator."""

    def __init__(
            self,
            poison_ratio: float = 0.5,
            target_label: int = 0,
            trigger_size: int = 3,
            seed: int = 42
    ):
        super().__init__(name="BackdoorAttack")
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.seed = seed

    def apply(self, dataset: Dataset) -> Dataset:
        return BackdoorDataset(
            dataset=dataset,
            poison_ratio=self.poison_ratio,
            target_label=self.target_label,
            trigger_size=self.trigger_size,
            seed=self.seed
        )
