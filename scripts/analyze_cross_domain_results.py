#!/usr/bin/env python3
"""
Analyze results from cross-domain noise experiments.
Generate 3x3 tables and visualizations for each experiment category.

Output:
- Console tables with accuracy scores
- CSV files with detailed results
- Heatmap visualizations (PNG)
- Summary report
"""

import os
import re
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")

# Configuration
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_results")
SNR_LEVELS = [-20, 0, 20]

# Experiment categories
CATEGORIES = {
    "baseline": {
        "name": "Baseline (Clean)",
        "pattern": r"baseline_clean",
        "is_matrix": False
    },
    "GG": {
        "name": "Same-Domain Gaussian",
        "pattern": r"GG_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "AA": {
        "name": "Same-Domain AudioSet",
        "pattern": r"AA_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "GA": {
        "name": "Cross-Domain Gaussian→AudioSet",
        "pattern": r"GA_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "AG": {
        "name": "Cross-Domain AudioSet→Gaussian",
        "pattern": r"AG_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "robust_gaussian": {
        "name": "Robustness (Gaussian→Clean)",
        "pattern": r"Robust_G(-?\d+)_Clean",
        "is_matrix": False
    },
    "robust_audioset": {
        "name": "Robustness (AudioSet→Clean)",
        "pattern": r"Robust_A(-?\d+)_Clean",
        "is_matrix": False
    }
}


def find_experiment_dirs():
    """Find all experiment directories in results."""
    experiment_dirs = {}

    for results_subdir in RESULTS_DIR.iterdir():
        if not results_subdir.is_dir():
            continue

        training_dir = results_subdir / "training"
        if not training_dir.exists():
            continue

        for exp_dir in training_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.endswith('.yaml'):
                experiment_dirs[exp_dir.name] = exp_dir

    return experiment_dirs


def parse_autrainer_dir_name(dir_name: str) -> dict:
    """Parse autrainer directory name to extract augmentation info.

    Format: dataset_model_optimizer_lr_batch_epoch_X_preprocessAug_trainAug_testAug_seed
    Examples:
    - TIMIT-sentencetype-16k_Cnn10-32k-T_Adam_0.001_64_epoch_10_None_CrossDomainNoise(0dB)_CrossDomainNoise(0dB)_0
    - TIMIT-sentencetype-16k_Cnn10-32k-T_Adam_0.001_64_epoch_10_None_GaussianNoise(0dB)_AudioSetNoise(-20dB)_0
    """
    info = {
        "train_aug_type": None,
        "test_aug_type": None,
        "train_snr": None,
        "test_snr": None
    }

    # Extract augmentation patterns
    # Pattern for noise augmentation: NoiseType(SNRdB) or NoiseType(-SNRdB)
    aug_pattern = r'(GaussianNoise|AudioSetNoise|CrossDomainNoise)\((-?\d+)dB\)'

    # Find all augmentation matches
    matches = list(re.finditer(aug_pattern, dir_name))

    if len(matches) >= 2:
        # First match is train augmentation, second is test
        info["train_aug_type"] = matches[0].group(1)
        info["train_snr"] = int(matches[0].group(2))
        info["test_aug_type"] = matches[1].group(1)
        info["test_snr"] = int(matches[1].group(2))
    elif len(matches) == 1:
        # Only one augmentation - determine if it's train or test based on position
        # In autrainer format: ..._None_TrainAug_TestAug_seed or ..._None_TrainAug_None_seed
        aug_str = matches[0].group(0)
        parts = dir_name.split('_')

        # Find the augmentation in parts
        for i, part in enumerate(parts):
            if aug_str in part or matches[0].group(1) in part:
                # Check if followed by None or _0 (seed)
                remaining = '_'.join(parts[i+1:])
                if remaining.startswith('None_') or remaining.startswith('0'):
                    # This is train aug, test is None
                    info["train_aug_type"] = matches[0].group(1)
                    info["train_snr"] = int(matches[0].group(2))
                elif i > 0 and 'None' in parts[i-1]:
                    # Previous was None, this could be test aug
                    info["test_aug_type"] = matches[0].group(1)
                    info["test_snr"] = int(matches[0].group(2))
                else:
                    # Default to train aug
                    info["train_aug_type"] = matches[0].group(1)
                    info["train_snr"] = int(matches[0].group(2))
                break

    return info


def get_best_metric(exp_dir: Path, metric_name: str = "f1") -> float:
    """Extract the best metric value from experiment directory."""
    import yaml

    # First try _test/test_holistic.yaml (autrainer's test results)
    test_results_file = exp_dir / "_test" / "test_holistic.yaml"
    if test_results_file.exists():
        try:
            with open(test_results_file) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and metric_name in data:
                    if isinstance(data[metric_name], dict):
                        return data[metric_name].get('all', None)
                    return data[metric_name]
        except Exception as e:
            print(f"  Warning: Could not read {test_results_file}: {e}")

    # Try _best/best_results.yaml
    best_results_file = exp_dir / "_best" / "best_results.yaml"
    if best_results_file.exists():
        try:
            with open(best_results_file) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and metric_name in data:
                    return data[metric_name]
        except Exception as e:
            print(f"  Warning: Could not read {best_results_file}: {e}")

    # Fallback to metrics.csv
    metrics_file = exp_dir / "metrics.csv"
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            if metric_name in df.columns:
                return df[metric_name].max()
            # Try variations
            for col in df.columns:
                if metric_name.lower() in col.lower():
                    return df[col].max()
        except Exception as e:
            print(f"  Warning: Could not read {metrics_file}: {e}")

    return None


def extract_results(experiment_dirs: dict) -> dict:
    """Extract results for all experiments."""
    results = {cat: {} for cat in CATEGORIES}

    for exp_name, exp_dir in experiment_dirs.items():
        accuracy = get_best_metric(exp_dir)
        if accuracy is None:
            continue

        # First try to match experiment_id patterns in config
        matched = False
        for cat_key, cat_info in CATEGORIES.items():
            match = re.search(cat_info["pattern"], exp_name)
            if match:
                if cat_info["is_matrix"]:
                    train_snr = int(match.group(1))
                    test_snr = int(match.group(2))
                    results[cat_key][(train_snr, test_snr)] = accuracy
                else:
                    if cat_key == "baseline":
                        results[cat_key]["clean"] = accuracy
                    else:
                        snr = int(match.group(1))
                        results[cat_key][snr] = accuracy
                matched = True
                break

        if matched:
            continue

        # If not matched, try parsing autrainer directory name
        info = parse_autrainer_dir_name(exp_name)

        if info["train_aug_type"] is None and info["test_aug_type"] is None:
            # No augmentation = baseline
            if "baseline" not in results or not results["baseline"]:
                results["baseline"]["clean"] = accuracy
            continue

        train_type = info["train_aug_type"]
        test_type = info["test_aug_type"]
        train_snr = info["train_snr"]
        test_snr = info["test_snr"]

        # Normalize augmentation types
        train_is_gaussian = train_type in ["GaussianNoise", "SNR_noise"] if train_type else False
        train_is_audioset = train_type in ["AudioSetNoise", "CrossDomainNoise"] if train_type else False
        test_is_gaussian = test_type in ["GaussianNoise", "SNR_noise"] if test_type else False
        test_is_audioset = test_type in ["AudioSetNoise", "CrossDomainNoise"] if test_type else False

        # Categorize experiment
        if train_type and test_type is None:
            # Robustness: train noisy, test clean
            if train_is_gaussian:
                results["robust_gaussian"][train_snr] = accuracy
            elif train_is_audioset:
                results["robust_audioset"][train_snr] = accuracy
        elif train_type and test_type:
            # Both train and test have augmentation
            if train_is_gaussian and test_is_gaussian:
                results["GG"][(train_snr, test_snr)] = accuracy
            elif train_is_audioset and test_is_audioset:
                results["AA"][(train_snr, test_snr)] = accuracy
            elif train_is_gaussian and test_is_audioset:
                results["GA"][(train_snr, test_snr)] = accuracy
            elif train_is_audioset and test_is_gaussian:
                results["AG"][(train_snr, test_snr)] = accuracy

    return results


def create_matrix_table(results: dict, category: str) -> pd.DataFrame:
    """Create a 3x3 matrix table from results."""
    matrix = np.full((len(SNR_LEVELS), len(SNR_LEVELS)), np.nan)

    for (train_snr, test_snr), accuracy in results.get(category, {}).items():
        try:
            i = SNR_LEVELS.index(train_snr)
            j = SNR_LEVELS.index(test_snr)
            matrix[i, j] = accuracy
        except ValueError:
            continue

    df = pd.DataFrame(
        matrix,
        index=[f"Train {snr}dB" for snr in SNR_LEVELS],
        columns=[f"Test {snr}dB" for snr in SNR_LEVELS]
    )
    return df


def print_matrix_table(df: pd.DataFrame, title: str):
    """Print a formatted matrix table to console."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    # Format values as percentages
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
        )

    print(formatted.to_string())


def save_heatmap(df: pd.DataFrame, title: str, filename: str):
    """Save a heatmap visualization."""
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(10, 8))

    # Convert to percentage
    df_pct = df * 100

    sns.heatmap(
        df_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Test SNR', fontsize=12)
    plt.ylabel('Train SNR', fontsize=12)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {filepath}")


def save_comparison_heatmap(results: dict, categories: list, filename: str):
    """Save a comparison heatmap for multiple categories."""
    if not HAS_MATPLOTLIB:
        return

    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(5*n_cats, 6))

    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        df = create_matrix_table(results, cat)
        df_pct = df * 100

        sns.heatmap(
            df_pct,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=50,
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={'label': 'Accuracy (%)'}
        )

        ax.set_title(CATEGORIES[cat]["name"], fontsize=12, fontweight='bold')
        ax.set_xlabel('Test SNR')
        ax.set_ylabel('Train SNR')

    plt.suptitle('Cross-Domain Noise Experiment Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison heatmap: {filepath}")


def save_robustness_barplot(results: dict, filename: str):
    """Save a bar plot for robustness experiments."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    x_labels = []
    accuracies = []
    colors = []

    # Baseline
    if "clean" in results.get("baseline", {}):
        x_labels.append("Baseline\n(Clean)")
        accuracies.append(results["baseline"]["clean"] * 100)
        colors.append('gray')

    # Gaussian robustness
    for snr in SNR_LEVELS:
        if snr in results.get("robust_gaussian", {}):
            x_labels.append(f"Gaussian\n{snr}dB→Clean")
            accuracies.append(results["robust_gaussian"][snr] * 100)
            colors.append('blue')

    # AudioSet robustness
    for snr in SNR_LEVELS:
        if snr in results.get("robust_audioset", {}):
            x_labels.append(f"AudioSet\n{snr}dB→Clean")
            accuracies.append(results["robust_audioset"][snr] * 100)
            colors.append('orange')

    if x_labels:
        bars = ax.bar(x_labels, accuracies, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Robustness Experiments: Train on Noisy → Test on Clean', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Baseline'),
            Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='Gaussian Noise'),
            Patch(facecolor='orange', edgecolor='black', alpha=0.8, label='AudioSet Noise')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved robustness plot: {filepath}")


def generate_summary_report(results: dict) -> str:
    """Generate a text summary report."""
    report = []
    report.append("=" * 80)
    report.append("CROSS-DOMAIN NOISE EXPERIMENTS - SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    # Baseline
    if "clean" in results.get("baseline", {}):
        report.append(f"\nBaseline (Clean train/test): {results['baseline']['clean']*100:.2f}%")

    # Matrix results
    for cat in ["GG", "AA", "GA", "AG"]:
        if results.get(cat):
            report.append(f"\n{'-'*40}")
            report.append(f"{CATEGORIES[cat]['name']}")
            report.append(f"{'-'*40}")

            df = create_matrix_table(results, cat)

            # Diagonal (matching conditions)
            diagonal = [df.iloc[i, i] for i in range(len(SNR_LEVELS)) if pd.notna(df.iloc[i, i])]
            if diagonal:
                report.append(f"  Diagonal mean (matched SNR): {np.mean(diagonal)*100:.2f}%")

            # Off-diagonal (mismatched conditions)
            off_diag = []
            for i in range(len(SNR_LEVELS)):
                for j in range(len(SNR_LEVELS)):
                    if i != j and pd.notna(df.iloc[i, j]):
                        off_diag.append(df.iloc[i, j])
            if off_diag:
                report.append(f"  Off-diagonal mean (mismatched SNR): {np.mean(off_diag)*100:.2f}%")

            # Best/worst
            valid = [(i, j, df.iloc[i, j]) for i in range(len(SNR_LEVELS))
                     for j in range(len(SNR_LEVELS)) if pd.notna(df.iloc[i, j])]
            if valid:
                best = max(valid, key=lambda x: x[2])
                worst = min(valid, key=lambda x: x[2])
                report.append(f"  Best: Train {SNR_LEVELS[best[0]]}dB, Test {SNR_LEVELS[best[1]]}dB = {best[2]*100:.2f}%")
                report.append(f"  Worst: Train {SNR_LEVELS[worst[0]]}dB, Test {SNR_LEVELS[worst[1]]}dB = {worst[2]*100:.2f}%")

    # Robustness results
    for cat in ["robust_gaussian", "robust_audioset"]:
        if results.get(cat):
            report.append(f"\n{'-'*40}")
            report.append(f"{CATEGORIES[cat]['name']}")
            report.append(f"{'-'*40}")
            for snr in SNR_LEVELS:
                if snr in results[cat]:
                    report.append(f"  SNR {snr}dB → Clean: {results[cat][snr]*100:.2f}%")

    report.append("\n" + "=" * 80)
    return "\n".join(report)


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Cross-Domain Noise Experiments - Results Analysis")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find experiments
    print("\nFinding experiment directories...")
    experiment_dirs = find_experiment_dirs()
    print(f"Found {len(experiment_dirs)} experiment directories")

    if not experiment_dirs:
        print("\nNo experiments found in results directory.")
        print("Run the training script first: sbatch train_cross_domain_full.sh")
        return 1

    # Extract results
    print("\nExtracting results...")
    results = extract_results(experiment_dirs)

    # Print and save matrix tables
    print("\n" + "=" * 60)
    print("RESULTS TABLES")
    print("=" * 60)

    for cat in ["GG", "AA", "GA", "AG"]:
        if results.get(cat):
            df = create_matrix_table(results, cat)
            print_matrix_table(df, CATEGORIES[cat]["name"])

            # Save to CSV
            csv_path = OUTPUT_DIR / f"{cat}_matrix.csv"
            df.to_csv(csv_path)
            print(f"  Saved CSV: {csv_path}")

            # Save heatmap
            save_heatmap(df, CATEGORIES[cat]["name"], f"{cat}_heatmap.png")

    # Comparison heatmaps
    print("\nGenerating comparison visualizations...")
    matrix_cats = [c for c in ["GG", "AA", "GA", "AG"] if results.get(c)]
    if len(matrix_cats) >= 2:
        save_comparison_heatmap(results, matrix_cats, "comparison_all.png")

        # Same-domain vs cross-domain
        same_domain = [c for c in ["GG", "AA"] if c in matrix_cats]
        cross_domain = [c for c in ["GA", "AG"] if c in matrix_cats]

        if same_domain:
            save_comparison_heatmap(results, same_domain, "comparison_same_domain.png")
        if cross_domain:
            save_comparison_heatmap(results, cross_domain, "comparison_cross_domain.png")

    # Robustness plot
    if results.get("robust_gaussian") or results.get("robust_audioset"):
        save_robustness_barplot(results, "robustness_barplot.png")

    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(results)
    print(report)

    # Save report
    report_path = OUTPUT_DIR / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report: {report_path}")

    # Save all results as JSON
    json_results = {}
    for cat, data in results.items():
        if data:
            json_results[cat] = {str(k): v for k, v in data.items()}

    json_path = OUTPUT_DIR / "all_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
