# # label_noise_augmentation.py

# import numpy as np
# import logging
# from typing import Optional, Sequence, Union

# import torch

# from .abstract_augmentation import AbstractAugmentation

# logger = logging.getLogger(__name__)


# class LabelNoise(AbstractAugmentation):
#     """
#     Unified Label Noise augmentation.
#     Supports three modes:
#         - mislabel: randomly replace label with another class
#         - smoothing: soft label distribution around true label
#         - random_noise: random soft distribution with true label dominant
#     """

#     def __init__(
#         self,
#         noise_level: float = 0.1,
#         labels: Optional[Sequence[Union[str, int]]] = None,
#         generator_seed: int = 0,
#         mode: str = "mislabel",   # mislabel | smoothing | random_noise
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.noise_level = noise_level
#         self.labels = list(labels) if labels is not None else None
#         self.generator_seed = generator_seed
#         self.mode = mode
#         self.rng = np.random.default_rng(generator_seed)

#     # ───────────────────────────── apply ─────────────────────────────

#     def apply(self, batch):
#         """
#         batch.target may be:
#             - int / string (single)
#             - list[int/string] (batch)
#             - torch.LongTensor (batch int class index)
#         """
#         y = batch.target

#         # Convert everything into tensor index format
#         y_tensor, num_classes = self._convert_labels_to_tensor(batch, y)

#         # Apply unified noise function
#         noisy = self._apply_noise(y_tensor, num_classes)

#         # Save back into batch
#         batch.target = noisy
#         return batch

#     # ───────────────────────── helper: convert to tensor ─────────────────────────

#     def _convert_labels_to_tensor(self, batch, y):
#         """
#         Convert target (string/int/list) to tensor of class indices.
#         Also retrieve dataset num_classes.
#         """
#         # dataset stores class names → map to integer
#         if hasattr(batch, "label_to_index"):
#             mapper = batch.label_to_index
#             num_classes = len(mapper)
#             if isinstance(y, list):
#                 y_tensor = torch.tensor([mapper[label] for label in y])
#             else:
#                 y_tensor = torch.tensor([mapper[y]])
#         else:
#             # Already integer
#             y_tensor = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
#             num_classes = batch.num_classes

#         return y_tensor, num_classes

#     # ───────────────────────── noise dispatcher ─────────────────────────

#     def _apply_noise(self, y_tensor, num_classes):
#         if self.mode == "mislabel":
#             return self._mislabel(y_tensor, num_classes)

#         if self.mode == "smoothing":
#             return self._smoothing(y_tensor, num_classes)

#         if self.mode == "random_noise":
#             return self._random_noise(y_tensor, num_classes)

#         raise ValueError(f"Unknown mode: {self.mode}")

#     # ───────────────────────── mode: mislabel ─────────────────────────

#     def _mislabel(self, targets: torch.Tensor, num_classes):
#         batch = len(targets)
#         noisy = targets.clone()

#         mask = torch.rand(batch) < self.noise_level
#         random_new = torch.randint(0, num_classes, (batch,))

#         # avoid selecting the same label
#         random_new = torch.where(random_new == targets, (random_new + 1) % num_classes, random_new)

#         noisy[mask] = random_new[mask]
#         return noisy

#     # ───────────────────────── mode: smoothing ─────────────────────────

#     def _smoothing(self, targets, num_classes):
#         alpha = self.noise_level
#         batch = len(targets)

#         soft = torch.full((batch, num_classes), alpha / (num_classes - 1))
#         soft[range(batch), targets] = 1 - alpha

#         return soft

#     # ───────────────────────── mode: random_noise ─────────────────────────

#     def _random_noise(self, targets, num_classes):
#         alpha = self.noise_level
#         batch = len(targets)

#         noise = torch.rand(batch, num_classes)
#         noise = noise / noise.sum(dim=1, keepdim=True)

#         soft = noise * alpha
#         soft[range(batch), targets] = 1 - alpha

#         return soft
