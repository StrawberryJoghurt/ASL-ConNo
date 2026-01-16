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
    Deterministic label-noise CrossEntropy on target vectors.

    Modes:
      - mislabel : deterministic per-sample flip with prob=noise_level, and deterministic wrong class
      - smoothing: label smoothing with alpha=noise_level
      - random   : deterministic per-sample random soft target mixed with one-hot by noise_level

    Notes:
      - Cache is stored on CPU to avoid GPU memory growth / fragmentation.
      - Cache key includes (sid, mode, noise_level, num_classes) to avoid cross-run contamination.
      - Supports either raw logits (default) or probabilities via inputs_are_probs flag.
    """

    def __init__(
        self,
        mode: str = "mislabel",
        noise_level: float = 0.1,
        inputs_are_probs: bool = False,   # IMPORTANT: set True only if you pass softmax(output)
        cache_on_cpu: bool = True,
        global_seed: int = 0,             # optional: change to get a different deterministic mapping
    ):
        super().__init__()
        self.mode = mode
        self.noise_level = float(noise_level)
        self.inputs_are_probs = bool(inputs_are_probs)
        self.cache_on_cpu = bool(cache_on_cpu)
        self.global_seed = int(global_seed)

        self.noise_cache = {}  # key -> noisy_target (CPU by default)
        print(f"[LOSS INIT] mode={mode}, noise={noise_level}, inputs_are_probs={inputs_are_probs}, cache_on_cpu={cache_on_cpu}")

    # --------------------------
    # helpers
    # --------------------------
    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _ensure_targets_1d_long(targets: torch.Tensor) -> torch.Tensor:
        # expects class indices
        if targets.dim() != 1:
            raise ValueError(f"targets must be 1D class indices (B,), got shape {tuple(targets.shape)}")
        if targets.dtype != torch.long:
            targets = targets.long()
        return targets

    def _make_noisy_target(
        self,
        clean_y: int,
        num_classes: int,
        mode: str,
        noise_level: float,
        device: torch.device,
        sample_idx: int,
    ) -> torch.Tensor:
        """
        Returns: (C,) target distribution (on CPU if cache_on_cpu else on device)
        """
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2 for label noise, got {num_classes}")

        nl = self._clamp01(noise_level)

        # where to allocate the returned vector
        out_device = torch.device("cpu") if self.cache_on_cpu else device

        # -------- mislabel --------
        if mode == "mislabel":
            # deterministic RNG streams (CPU) based on sample_idx + global_seed
            sid = int(sample_idx)
            base = sid + 1000003 * self.global_seed  # simple mix

            # choose wrong class deterministically
            g1 = torch.Generator()
            g1.manual_seed(base)

            # build wrong-class list on CPU for simplicity (small C)
            # (this avoids any GPU involvement in sampling)
            wrong_classes = list(range(num_classes))
            wrong_classes.remove(int(clean_y))

            j = torch.randint(len(wrong_classes), (1,), generator=g1).item()
            noisy_class = int(wrong_classes[j])

            # deterministic flip decision
            g2 = torch.Generator()
            g2.manual_seed(base + 13)
            u = torch.rand((), generator=g2).item()

            y = torch.zeros(num_classes, device=out_device, dtype=torch.float32)
            if u < nl:
                y[noisy_class] = 1.0
            else:
                y[int(clean_y)] = 1.0
            return y

        # -------- smoothing --------
        elif mode == "smoothing":
            # classic label smoothing: alpha = nl
            # y_k = 1-nl for true class, nl/(C-1) otherwise
            smooth = nl / (num_classes - 1)
            y = torch.full((num_classes,), smooth, device=out_device, dtype=torch.float32)
            y[int(clean_y)] = 1.0 - nl
            return y

        # -------- random soft noise --------
        elif mode == "random":
            # deterministic random vector
            g = torch.Generator(device=device)
            g.manual_seed(int(sample_idx))

            noise = torch.rand(num_classes, device=device, generator=g)
            noise = noise / noise.sum()

            y_onehot = torch.zeros(num_classes, device=device)
            y_onehot[int(clean_y)] = 1.0

            # use clamped nl (safety, no effect for your normal configs)
            y_noisy = (1 - nl) * y_onehot + nl * noise

            # respect cache_on_cpu
            if out_device.type == "cpu":
                y_noisy = y_noisy.detach().to("cpu")

            return y_noisy


    # --------------------------
    # forward
    # --------------------------
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        inputs: (B, C)
          - if inputs_are_probs=False: raw logits
          - if inputs_are_probs=True : probabilities (already softmaxed)
        targets: (B,) class indices
        idx: (B,) sample indices (must be stable per sample across epochs)
        returns: (B,) per-sample loss
        """
        if inputs.dim() != 2:
            raise ValueError(f"inputs must be 2D (B,C), got shape {tuple(inputs.shape)}")

        B, C = inputs.shape
        device = inputs.device

        targets = self._ensure_targets_1d_long(targets)

        if idx.dim() != 1 or idx.numel() != B:
            raise ValueError(f"idx must be 1D of length B. Got shape {tuple(idx.shape)} with B={B}")

        # clean labels in eval or when disabled
        if (not self.training) or (self._clamp01(self.noise_level) == 0.0) or (self.mode == "none"):
            # NOTE: this assumes raw logits. If you pass probabilities in eval, set inputs_are_probs=True and use NLL below.
            if not self.inputs_are_probs:
                return F.cross_entropy(inputs, targets, reduction="none")
            else:
                log_probs = torch.log(inputs.clamp_min(1e-12))
                return F.nll_loss(log_probs, targets, reduction="none")

        mode = self.mode
        nl = self._clamp01(self.noise_level)

        # build noisy targets (cache)
        noisy_list = []
        idx_cpu = idx.detach().to("cpu")  # safe for int conversion
        targets_cpu = targets.detach().to("cpu")

        for clean_y_t, sid_t in zip(targets_cpu, idx_cpu):
            sid = int(sid_t.item())
            clean_y = int(clean_y_t.item())

            cache_key = (sid, mode, float(nl), int(C))
            if cache_key in self.noise_cache:
                noisy_list.append(self.noise_cache[cache_key])
                continue

            noisy_t = self._make_noisy_target(
                clean_y=clean_y,
                num_classes=C,
                mode=mode,
                noise_level=nl,
                device=device,
                sample_idx=sid,
            )

            # store in cache (already on CPU if cache_on_cpu=True)
            self.noise_cache[cache_key] = noisy_t
            noisy_list.append(noisy_t)

        noisy_targets = torch.stack(noisy_list, dim=0)  # (B,C) on CPU if cache_on_cpu
        if noisy_targets.device != device:
            noisy_targets = noisy_targets.to(device, non_blocking=True)

        # compute log-probs
        if not self.inputs_are_probs:
            log_probs = F.log_softmax(inputs, dim=1)
        else:
            # inputs are probabilities
            log_probs = torch.log(inputs.clamp_min(1e-12))

        # CE(p, q) = - sum_c q_c * log p_c
        loss = -(noisy_targets * log_probs).sum(dim=1)

        return loss
