from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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
    Deterministic label-noise CrossEntropy:
    - mislabel: fixed wrong-class per sample
    - smoothing: deterministic
    - random: fixed soft target per sample
    """

    def __init__(self, mode="mislabel", noise_level=0.1):
        super().__init__()
        self.mode = mode
        self.noise_level = noise_level
        self.noise_cache = {}          # sample_idx → noisy_target
        print(f"[LOSS INIT] mode={mode}, noise={noise_level}")

    # --------------------------
    # utils
    # --------------------------
    def _one_hot(self, y, num_classes):
        y_onehot = torch.zeros((y.size(0), num_classes), device=y.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        return y_onehot

    # --------------------------
    # deterministic noise maker
    # --------------------------
    def _make_noisy_target(self, clean_y, num_classes, mode, noise_level, device, sample_idx):
        """
        clean_y: int (scalar tensor)
        return shape => (num_classes,) soft or hard distribution
        """

        # -------- mislabel --------
        if mode == "mislabel":
            all_cls = torch.arange(num_classes, device=device)
            wrong = all_cls[all_cls != clean_y]

            # deterministic: use sample_idx as RNG seed
            g = torch.Generator(device=device)
            g.manual_seed(int(sample_idx))

            noisy_class = wrong[torch.randint(len(wrong), (1,), generator=g)]
            noisy_onehot = torch.zeros(num_classes, device=device)
            noisy_onehot[noisy_class] = 1.0
            return noisy_onehot

        # -------- smoothing --------
        elif mode == "smoothing":
            smooth = noise_level / (num_classes - 1)
            y_smooth = torch.full((num_classes,), smooth, device=device)
            y_smooth[clean_y] = 1.0 - noise_level
            return y_smooth

        # -------- random soft noise --------
        elif mode == "random":
            # deterministic random vector
            g = torch.Generator(device=device)
            g.manual_seed(int(sample_idx))

            noise = torch.rand(num_classes, device=device, generator=g)
            noise = noise / noise.sum()

            y_onehot = torch.zeros(num_classes, device=device)
            y_onehot[clean_y] = 1.0

            y_noisy = (1 - noise_level) * y_onehot + noise_level * noise
            return y_noisy

        else:
            raise ValueError(f"Unknown noise mode {mode}")

    # --------------------------
    # main forward
    # --------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, idx: torch.Tensor):
        """
        logits: (B, C)
        targets: (B,)
        idx: (B,) sample indices
        """
        num_classes = logits.size(1)
        device = logits.device

        # -------------------------------------
        # eval mode → always clean labels
        # -------------------------------------
        if not self.training or self.noise_level == 0 or self.mode == "none":
            return F.cross_entropy(logits, targets, reduction="none")

        mode = self.mode
        nl = self.noise_level

        # -------------------------------------
        # generate deterministic noisy targets
        # -------------------------------------
        noisy_list = []
        for clean_y, sid in zip(targets, idx):
            sid = int(sid)

            # cache hit
            if sid in self.noise_cache:
                noisy_list.append(self.noise_cache[sid])
                continue

            # create new
            noisy_t = self._make_noisy_target(
                clean_y=int(clean_y),
                num_classes=num_classes,
                mode=mode,
                noise_level=nl,
                device=device,
                sample_idx=sid,
            )

            self.noise_cache[sid] = noisy_t
            noisy_list.append(noisy_t)

        noisy_targets = torch.stack(noisy_list)    # shape (B, C)

        # -------------------------------------
        # compute cross entropy manually
        # CE(p, q) = - sum( q * log_softmax(p) )
        # -------------------------------------
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(noisy_targets * log_probs).sum(dim=1)

        return loss