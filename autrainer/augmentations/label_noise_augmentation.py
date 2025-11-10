import torch
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LabelNoise:
    """
    On-the-fly label noise augmentation for autrainer.
    Supports two modes:
      1. probabilistic — each label has p chance to be replaced (approx. noise_rate% labels)
      2. fixed — exactly noise_rate% of labels are replaced per batch

    Args:
        noise_rate (float): proportion of labels to corrupt (0–1)
        num_classes (int): number of target classes
        seed (int): random seed for reproducibility
        mode (str): "probabilistic" or "fixed"
    """

    def __init__(self, noise_rate=0.1, num_classes=None, seed=0, mode="fixed"):
        assert 0 <= noise_rate <= 1, "noise_rate must be in [0,1]"
        assert mode in ["probabilistic", "fixed"], "mode must be 'probabilistic' or 'fixed'"

        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.seed = seed
        self.mode = mode

        # Independent RNGs (to make this deterministic given seed)
        self.rng = np.random.default_rng(seed)
        self.rng_py = random.Random(seed)

        logger.info(f"[LabelNoise] Initialized with mode={mode}, rate={noise_rate}, seed={seed}")

    def __call__(self, batch):
        """Apply label noise on-the-fly during data loading."""
        inputs, labels = batch
        labels = np.array(labels)
        n = len(labels)

        if self.num_classes is None:
            raise ValueError("num_classes must be provided to apply label noise")

        noisy_labels = labels.copy()

        if self.mode == "probabilistic":
            # Each label has a p chance to be replaced
            for i, y in enumerate(labels):
                if self.rng_py.random() < self.noise_rate:
                    new_label = self.rng_py.randint(0, self.num_classes - 1)
                    while new_label == y:
                        new_label = self.rng_py.randint(0, self.num_classes - 1)
                    noisy_labels[i] = new_label

            logger.debug(f"[LabelNoise] Applied probabilistic noise, rate={self.noise_rate}")

        elif self.mode == "fixed":
            # Replace exactly k = noise_rate * n labels
            k = int(n * self.noise_rate)
            if k > 0:
                noisy_indices = self.rng.choice(n, size=k, replace=False)
                for idx in noisy_indices:
                    y = labels[idx]
                    new_label = self.rng.integers(0, self.num_classes)
                    while new_label == y:
                        new_label = self.rng.integers(0, self.num_classes)
                    noisy_labels[idx] = new_label

            logger.debug(f"[LabelNoise] Applied fixed noise, replaced {k}/{n} labels")

        return torch.tensor(inputs), torch.tensor(noisy_labels)

# # small test
# batch = (torch.randn(5, 3), [0, 1, 2, 3, 4])
# ln1 = LabelNoise(noise_rate=0.4, num_classes=5, seed=2, mode="fixed")
# ln2 = LabelNoise(noise_rate=0.4, num_classes=5, seed=2, mode="fixed")

# _, y1 = ln1(batch)
# _, y2 = ln2(batch)

# print(y1, y2)
# assert torch.equal(y1, y2)