from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .utils import assert_nonzero_frequency


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.datasets import AbstractDataset


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.CrossEntropyLoss.forward`.

        Converts the targets to `long` if it is a 1D tensor.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        if y.ndim == 1:
            y = y.long()
        return super().forward(x, y)


class BalancedCrossEntropyLoss(CrossEntropyLoss):
    def setup(self, data: "AbstractDataset") -> None:
        """Calculate balanced weights for the dataset based on the target
        frequency in the training set.

        Args:
            data: Instance of the dataset.
        """
        frequency = (
            data.df_train[data.target_column]
            .map(data.target_transform)
            .value_counts()
            .sort_index()
            .values
        )

        assert_nonzero_frequency(frequency, len(data.target_transform))
        weight = torch.tensor(1 / frequency, dtype=torch.float32)
        self.weight = weight * len(weight) / weight.sum()


class WeightedCrossEntropyLoss(BalancedCrossEntropyLoss):
    def __init__(
        self,
        class_weights: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Wrapper for `torch.nn.CrossEntropyLoss` with manual class weights.

        The class weights are automatically normalized to sum up to the number
        of classes.

        Args:
            class_weights: Dictionary with class weights corresponding to the
                target labels and their respective weights.
            **kwargs: Additional keyword arguments passed to
                `torch.nn.CrossEntropyLoss`.
        """
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate the class weights based on the provided dictionary.

        Args:
            data: Instance of the dataset.
        """
        values = []

        for label in data.target_transform.labels:
            if label not in self.class_weights:
                raise ValueError(f"Missing class weight for label '{label}'.")
            values.append(self.class_weights[label])

        assert_nonzero_frequency(np.array(values), len(data.target_transform))
        weight = torch.tensor(values, dtype=torch.float32)
        self.weight = weight * len(weight) / weight.sum()


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.BCEWithLogitsLoss.forward`.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        return super().forward(x, y.float())


class BalancedBCEWithLogitsLoss(BCEWithLogitsLoss):
    weights_buffer: torch.Tensor

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Balanced version of `torch.nn.BCEWithLogitsLoss`.

        `pos_weight` is not supported, as the weights are calculated based on
        the target frequency in the training set.

        Args:
            weight: A manual rescaling weight given to the positive class.
                Defaults to None.
            reduction: Specifies the reduction to apply to the output. Defaults
                to 'mean'.
        """
        super().__init__(weight=weight, reduction=reduction)

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate balanced weights for the dataset based on the target
        frequency in the training set.

        Args:
            data: Instance of the dataset.
        """

        def encode(x: pd.Series) -> List[int]:
            return data.target_transform(x.to_list()).tolist()

        frequency = (
            pd.DataFrame(
                data.df_train[data.target_column].apply(encode, axis=1).to_list(),
                columns=data.target_transform.labels,
            )
            .sum(axis=0)
            .values
        )

        assert_nonzero_frequency(frequency, len(data.target_transform))
        weight = torch.tensor(1 / frequency, dtype=torch.float32)
        weight = weight * len(weight) / weight.sum()
        self.register_buffer("weights_buffer", weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.BCEWithLogitsLoss.forward` with balanced
        weights.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        return super().forward(x, y) * self.weights_buffer.expand_as(y)


class WeightedBCEWithLogitsLoss(BalancedBCEWithLogitsLoss):
    def __init__(
        self,
        class_weights: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Wrapper for `torch.nn.BCEWithLogitsLoss` with manual class weights.

        The class weights are automatically normalized to sum up to the number
        of classes.

        Args:
            class_weights: Dictionary with class weights corresponding to the
                target labels and their respective weights.
            **kwargs: Additional keyword arguments passed to
                `torch.nn.BCEWithLogitsLoss`.
        """
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate the class weights based on the provided dictionary.

        Args:
            data: Instance of the dataset.
        """
        values = []

        for label in data.target_transform.labels:
            if label not in self.class_weights:
                raise ValueError(f"Missing class weight for label '{label}'.")
            values.append(self.class_weights[label])
        assert_nonzero_frequency(np.array(values), len(data.target_transform))
        weight = torch.tensor(values, dtype=torch.float32)
        weight = weight * len(weight) / weight.sum()
        self.register_buffer("weights_buffer", weight)



class AutoCrossEntropyLoss(torch.nn.Module):
    """
    Unified loss that applies label distortion (mislabel, smoothing, random_noise)
    automatically inside the loss function.
    """

    def __init__(self, mode="mislabel", noise_level=0.1):
        super().__init__()
        self.mode = mode
        self.noise_level = noise_level
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def _one_hot(self, y, num_classes):
        y_onehot = torch.zeros((y.size(0), num_classes), device=y.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        return y_onehot

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        if y.ndim == 1:
            y = y.long()

        num_classes = x.size(1)
        y_onehot = self._one_hot(y, num_classes)

        # 0) No noise
        if self.noise_level <= 0 or self.mode == "none":
            log_prob = torch.log(x + 1e-8)
            return -(y_onehot * log_prob).sum(dim=1)

        # 1) Smoothing
        if self.mode == "smoothing":
            smooth = self.noise_level / (num_classes - 1)
            y_smooth = torch.full_like(y_onehot, smooth)
            y_smooth.scatter_(1, y.unsqueeze(1), 1.0 - self.noise_level)
            return -(y_smooth * torch.log(x + 1e-8)).sum(dim=1)

        # 2) Mislabel
        elif self.mode == "mislabel":
            B = y.size(0)
            mask = (torch.rand(B, device=y.device) < self.noise_level)
            y_noisy = y.clone()

            for i in range(B):
                if mask[i]:
                    all_cls = torch.arange(num_classes, device=y.device)
                    wrong = all_cls[all_cls != y[i]]
                    y_noisy[i] = wrong[torch.randint(len(wrong), (1,))]
            return self.ce(x, y_noisy)

        # 3) Random noise
        elif self.mode == "random":
            noise = torch.rand_like(y_onehot)
            noise = noise / noise.sum(dim=1, keepdim=True)
            y_noisy = (1 - self.noise_level) * y_onehot + self.noise_level * noise
            return -(y_noisy * torch.log(x + 1e-8)).sum(dim=1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward_with_debug(self, x: torch.Tensor, y: torch.Tensor):
        """
        Returns:
            loss (scalar)
            noisy_label (tensor)
        """
        if y.ndim == 1:
            y = y.long()

        num_classes = x.size(1)
        y_onehot = self._one_hot(y, num_classes)

        # 0) no noise
        if self.noise_level <= 0 or self.mode == "none":
            log_prob = torch.log(x + 1e-8)
            loss = -(y_onehot * log_prob).sum(dim=1).mean()
            return loss, y

        # 1) Smoothing
        if self.mode == "smoothing":
            smooth = self.noise_level / (num_classes - 1)
            y_smooth = torch.full_like(y_onehot, smooth)
            y_smooth.scatter_(1, y.unsqueeze(1), 1.0 - self.noise_level)
            loss = -(y_smooth * torch.log(x + 1e-8)).sum(dim=1).mean()
            return loss, y_smooth

        # 2) Mislabel
        elif self.mode == "mislabel":
            B = y.size(0)
            mask = (torch.rand(B, device=y.device) < self.noise_level)
            y_noisy = y.clone()
            for i in range(B):
                if mask[i]:
                    all_cls = torch.arange(num_classes, device=y.device)
                    wrong = all_cls[all_cls != y[i]]
                    y_noisy[i] = wrong[torch.randint(len(wrong), (1,))]
            loss = self.ce(x, y_noisy).mean()
            return loss, y_noisy

        # 3) Random noise
        elif self.mode == "random_noise":
            noise = torch.rand_like(y_onehot)
            noise = noise / noise.sum(dim=1, keepdim=True)
            y_noisy = (1 - self.noise_level) * y_onehot + self.noise_level * noise
            loss = -(y_noisy * torch.log(x + 1e-8)).sum(dim=1).mean()
            return loss, y_noisy

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
