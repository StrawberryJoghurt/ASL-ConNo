import torch
import numpy as np
import logging
from typing import Optional, Sequence

from .abstract_augmentation import AbstractAugmentation

logger = logging.getLogger(__name__)

class LabelNoise(AbstractAugmentation):
    """
    LabelNoise augmentation for string targets (TIMIT dialect style).
    Adds noise directly on label strings before encoding.
    """

    def __init__(
        self,
        noise_rate: float = 0.1,
        labels: Optional[Sequence[str]] = None,  # all possible label names
        generator_seed: int = 0,
        mode: str = "fixed",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_rate = noise_rate
        self.labels = list(labels) if labels is not None else None
        self.generator_seed = generator_seed
        self.mode = mode
        self.rng = np.random.default_rng(self.generator_seed)

    def apply(self, batch):
        """Replace some label strings randomly with others."""
        y = batch.target

        # single sample
        if isinstance(y, str):
            if self.labels and self.rng.random() < self.noise_rate:
                new_label = self.rng.choice(self.labels)
                while new_label == y:
                    new_label = self.rng.choice(self.labels)
                batch.target = new_label
                logger.debug(f"[LabelNoise] single {y} → {new_label}")
            return batch

        # batch mode
        if isinstance(y, (list, tuple)):
            n = len(y)
            y = np.array(y)
            noisy_y = y.copy()
            k = int(n * self.noise_rate)
            if k > 0 and self.labels:
                idxs = self.rng.choice(n, size=k, replace=False)
                for idx in idxs:
                    new_label = self.rng.choice(self.labels)
                    while new_label == y[idx]:
                        new_label = self.rng.choice(self.labels)
                    noisy_y[idx] = new_label
            batch.target = list(noisy_y)
        return batch
