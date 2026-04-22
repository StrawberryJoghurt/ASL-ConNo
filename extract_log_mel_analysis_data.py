#!/usr/bin/env python3
"""Plot figures corresponding to log_mel_analysis.ipynb.

Included figure groups:
1) Train with wind / bell / pink noise and test on multiple noise sets.
2) Train with wind-sub (4 mel bands) and test on multiple noise sets.
3) "Normalized Audio Intensity - Frequency" (wind vs pink) from AudioSet.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

SNR_LIST = [0, 5, 10, 15]
TEST_TYPE_LIST = list(range(9))
MAPPING = {
    0: "Wild animals",
    1: "Domestic animals, pets",
    2: "Bell",
    3: "Alarm",
    4: "Wind",
    5: "Water",
}


def _read_f1_from_yaml(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return float(data["f1"]["all"])


def _collect_from_template(
    template: str,
    snr_list: list[int],
    test_type_list: list[int],
    strict: bool,
    **template_kwargs: Any,
) -> tuple[dict[int, list[float]], list[str]]:
    result: dict[int, list[float]] = {}
    missing_files: list[str] = []

    for snr in snr_list:
        values: list[float] = []
        for test_type in test_type_list:
            path = Path(
                template.format(
                    snr=snr,
                    test_type=test_type,
                    **template_kwargs,
                )
            )
            if not path.exists():
                missing_files.append(str(path))
                continue
            values.append(_read_f1_from_yaml(path))

        if values:
            result[snr] = values

    if strict and missing_files:
        msg = "\n".join(missing_files[:20])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} files. First 20 missing:\n{msg}"
        )

    return result, missing_files


def collect_class1_train_noise_vs_test(repo_root: Path, strict: bool) -> dict[str, Any]:
    """Class 1: train with wind/bell/pink, evaluate on test noise sets."""
    templates = {
        "wind": (
            "{repo_root}/results/pink_wind/training/"
            "TIMIT-sentencetype-16k_Cnn10-32k-T_Adam_0.001_64_epoch_4_None_"
            "AudioSet4({snr})_AudioSet{snr}_0/_test{test_type}/test_holistic.yaml"
        ),
        "pink": (
            "{repo_root}/results/pink_wind/training/"
            "TIMIT-sentencetype-16k_Cnn10-32k-T_Adam_0.001_64_epoch_4_None_"
            "AudioSet9({snr})_AudioSet{snr}_0/_test{test_type}/test_holistic.yaml"
        ),
        "bell": (
            "{repo_root}/results/bell/training/"
            "TIMIT-sentencetype-16k_Cnn10-32k-T_Adam_0.001_64_epoch_4_None_"
            "AudioSet5({snr})_AudioSet{snr}_0/_test{test_type}/test_holistic.yaml"
        ),
    }

    out: dict[str, Any] = {"data": {}, "missing_files": {}}

    for noise_name, template in templates.items():
        data, missing = _collect_from_template(
            template=template,
            snr_list=SNR_LIST,
            test_type_list=TEST_TYPE_LIST,
            strict=strict,
            repo_root=repo_root,
        )
        out["data"][noise_name] = data
        out["missing_files"][noise_name] = missing

    return out


def collect_class2_wind_sub_bands(repo_root: Path, strict: bool) -> dict[str, Any]:
    """Class 2: wind noise split into four mel-band groups."""
    template = (
        "{repo_root}/results/wind_sub/training/"
        "TIMIT-sentencetype-16k_Cnn10-32k-T_Adam_0.001_64_epoch_4_None_"
        "AudioSet4-{band}-({snr})_AudioSet{snr}_0/_test{test_type}/test_holistic.yaml"
    )
    out: dict[str, Any] = {"data": {}, "missing_files": {}}

    for band in [0, 1, 2, 3]:
        data, missing = _collect_from_template(
            template=template,
            snr_list=SNR_LIST,
            test_type_list=TEST_TYPE_LIST,
            strict=strict,
            repo_root=repo_root,
            band=band,
        )
        out["data"][str(band)] = data
        out["missing_files"][str(band)] = missing

    return out


def _compute_normalized_curve(data: dict[str, Any], label_idx: int) -> dict[str, Any]:
    import torch

    mel_indices = data["audio_index"]["train"][label_idx]
    mel_list = [data["ds_logmel"]["train"]["log_mel"][i] for i in mel_indices]
    mel = torch.tensor(mel_list, dtype=torch.float32)

    if mel.size(-1) <= 23:
        raise ValueError(f"Invalid mel length: {mel.size(-1)} (must be > 23).")

    # Same formula as in the notebook:
    # avg[i][j] = mean(10 ** (mel[i][j][:-23] / 10))
    avg = torch.pow(10.0, mel[:, :, :-23] / 10.0).mean(dim=2)  # [n_samples, 64]
    curve = avg.mean(dim=0)  # [64]
    curve_norm = curve / curve.max()

    return {
        "label_idx": label_idx,
        "n_samples": int(mel.shape[0]),
        "mean_linear_intensity": [float(x) for x in curve.tolist()],
        "normalized_curve": [float(x) for x in curve_norm.tolist()],
    }


def collect_class3_normalized_intensity_frequency(repo_root: Path) -> dict[str, Any]:
    """Class 3: normalized audio intensity vs frequency (wind vs pink)."""
    pt_path = repo_root / "data/AudioSet/AudioSet.pt"
    if not pt_path.exists():
        return {
            "source_file": str(pt_path),
            "error": "AudioSet.pt not found",
        }

    try:
        import torch
    except ImportError:
        return {
            "source_file": str(pt_path),
            "error": "PyTorch is not installed; class3 extraction skipped",
        }
    print(pt_path)
    data = torch.load(pt_path, map_location="cpu")

    return {
        "source_file": str(pt_path),
        "wind": _compute_normalized_curve(data, label_idx=4),
        "pink": _compute_normalized_curve(data, label_idx=9),
    }


def _plot_grouped_bar(
    data_by_snr: dict[int, list[float]],
    title: str,
    labels: list[str],
    snr_list: list[int],
    max_groups: int = 6,
) -> plt.Figure:
    x = np.array(snr_list, dtype=float)
    x_pos = np.arange(len(x))
    values = np.array([data_by_snr[snr][:max_groups] for snr in snr_list], dtype=float)

    n_groups = values.shape[1]
    bar_width = 0.12
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i in range(n_groups):
        ax.bar(
            x_pos + (i - (n_groups - 1) / 2) * bar_width,
            values[:, i],
            width=bar_width,
            color=colors[i % len(colors)],
            label=labels[i],
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.set_xticks(x_pos, [int(v) for v in x])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def plot_class1(class1_data: dict[str, Any]) -> dict[str, plt.Figure]:
    labels = [MAPPING[i] for i in sorted(MAPPING.keys())]
    figures: dict[str, plt.Figure] = {}

    for train_noise, title in [
        ("wind", "Train with Wind noise: F1 vs SNR"),
        ("pink", "Train with Pink noise: F1 vs SNR"),
        ("bell", "Train with Bell noise: F1 vs SNR"),
    ]:
        data_by_snr = class1_data["data"][train_noise]
        snr_list = sorted(int(k) for k in data_by_snr.keys())
        data_by_snr_int_key = {int(k): v for k, v in data_by_snr.items()}
        figures[train_noise] = _plot_grouped_bar(
            data_by_snr=data_by_snr_int_key,
            title=title,
            labels=labels,
            snr_list=snr_list,
            max_groups=min(6, len(labels)),
        )

    return figures


def plot_class2(class2_data: dict[str, Any]) -> dict[str, plt.Figure]:
    labels = [MAPPING[i] for i in sorted(MAPPING.keys())]
    figures: dict[str, plt.Figure] = {}
    band_ranges = {
        "0": "Band 0-15",
        "1": "Band 16-31",
        "2": "Band 32-47",
        "3": "Band 48-63",
    }

    for band in ["0", "1", "2", "3"]:
        data_by_snr = class2_data["data"][band]
        snr_list = sorted(int(k) for k in data_by_snr.keys())
        data_by_snr_int_key = {int(k): v for k, v in data_by_snr.items()}
        figures[band] = _plot_grouped_bar(
            data_by_snr=data_by_snr_int_key,
            title=f"{band_ranges[band]}: F1 vs SNR",
            labels=labels,
            snr_list=snr_list,
            max_groups=min(6, len(labels)),
        )

    return figures


def plot_class3(class3_data: dict[str, Any]) -> plt.Figure | None:
    if "error" in class3_data:
        print("class3 skipped:", class3_data["error"])
        return None

    wind = class3_data["wind"]["normalized_curve"]
    pink = class3_data["pink"]["normalized_curve"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(wind, label="Wind")
    ax.plot(pink, label="Pink Noise")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Normalized Audio Intensity - Frequency")
    ax.set_xlabel("Sample frequency")
    ax.set_ylabel("Normalized intensity")
    fig.tight_layout()
    return fig


def _save_figures(figures: dict[str, plt.Figure], save_dir: Path, prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        out = save_dir / f"{prefix}_{name}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("Saved:", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot figures from log_mel_analysis.ipynb workflow."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save PNG figures.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error when any expected YAML file is missing.",
    )
    parser.add_argument(
        "--skip-class3",
        action="store_true",
        help="Skip class3 extraction (AudioSet.pt + torch).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plotting windows; use with --save-dir for headless runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    class1_data = collect_class1_train_noise_vs_test(repo_root=repo_root, strict=args.strict)
    class2_data = collect_class2_wind_sub_bands(repo_root=repo_root, strict=args.strict)

    class1_figs = plot_class1(class1_data)
    class2_figs = plot_class2(class2_data)

    class3_fig = None
    if not args.skip_class3:
        class3_data = collect_class3_normalized_intensity_frequency(repo_root=repo_root)
        class3_fig = plot_class3(class3_data)

    if args.save_dir is not None:
        save_dir = args.save_dir
        if not save_dir.is_absolute():
            save_dir = repo_root / save_dir
        _save_figures(class1_figs, save_dir, "class1")
        _save_figures(class2_figs, save_dir, "class2")
        if class3_fig is not None:
            out = save_dir / "class3_normalized_intensity_frequency.png"
            class3_fig.savefig(out, dpi=180, bbox_inches="tight")
            print("Saved:", out)

    print(
        "class1 missing files:",
        sum(len(v) for v in class1_data["missing_files"].values()),
    )
    print(
        "class2 missing files:",
        sum(len(v) for v in class2_data["missing_files"].values()),
    )
    print("Plotted figures:", len(class1_figs) + len(class2_figs) + (1 if class3_fig else 0))

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
