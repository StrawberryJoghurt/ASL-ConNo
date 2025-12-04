#!/usr/bin/env python3
"""
Analyze results from cross-domain noise experiments.
Generate 4x4 tables and visualizations for each experiment category.

NEW Structure (v2):
- Ideal: Clean→Clean (upper bound, best possible performance)
- Baseline Degradation: Clean Train + Noisy Test (shows how model breaks)
- Same-domain: GG, AA (train and test with same noise type)
- Cross-domain: GA, AG (train with one noise, test with another)
- Robustness: Train noisy → Test clean

SNR Levels: -5dB, 0dB, 10dB, 20dB (excluding -20dB)
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

# NEW: Updated SNR levels (excluding -20dB)
SNR_LEVELS = [-5, 0, 10, 20]

# Experiment categories
CATEGORIES = {
    "ideal": {
        "name": "Ideal (Clean→Clean)",
        "pattern": r"baseline_clean",
        "is_matrix": False
    },
    "baseline_gaussian": {
        "name": "Baseline Degradation (Clean→Gaussian)",
        "pattern": r"CleanTrain_Gaussian(-?\d+)",
        "is_matrix": False
    },
    "baseline_audioset": {
        "name": "Baseline Degradation (Clean→AudioSet)",
        "pattern": r"CleanTrain_AudioSet(-?\d+)",
        "is_matrix": False
    },
    "GG": {
        "name": "Same-Domain Gaussian (G→G)",
        "pattern": r"GG_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "AA": {
        "name": "Same-Domain AudioSet (A→A)",
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
    """Find all experiment directories in results.

    Returns dict mapping experiment_id (top-level dir name) to the actual
    experiment directory inside training/.
    """
    experiment_dirs = {}

    for results_subdir in RESULTS_DIR.iterdir():
        if not results_subdir.is_dir():
            continue

        training_dir = results_subdir / "training"
        if not training_dir.exists():
            continue

        for exp_dir in training_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.endswith('.yaml'):
                # Use the top-level directory name as the experiment ID
                # (e.g., "AA_train0_test0" instead of "TIMIT-sentencetype-...")
                experiment_dirs[results_subdir.name] = exp_dir

    return experiment_dirs


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

        # Match experiment patterns
        for cat_key, cat_info in CATEGORIES.items():
            match = re.search(cat_info["pattern"], exp_name)
            if match:
                if cat_info["is_matrix"]:
                    train_snr = int(match.group(1))
                    test_snr = int(match.group(2))
                    # Skip -20dB results
                    if train_snr == -20 or test_snr == -20:
                        continue
                    results[cat_key][(train_snr, test_snr)] = accuracy
                else:
                    if cat_key == "ideal":
                        results[cat_key]["clean"] = accuracy
                    else:
                        snr = int(match.group(1))
                        # Skip -20dB results
                        if snr == -20:
                            continue
                        results[cat_key][snr] = accuracy
                break

    return results


def create_matrix_table(results: dict, category: str) -> pd.DataFrame:
    """Create a 4x4 matrix table from results."""
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
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    # Format values as percentages
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
        )

    print(formatted.to_string())


def save_heatmap(df: pd.DataFrame, title: str, filename: str, baseline_values: dict = None):
    """Save a heatmap visualization with optional baseline comparison."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

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
        cbar_kws={'label': 'Accuracy (%)'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Test SNR', fontsize=12)
    ax.set_ylabel('Train SNR', fontsize=12)

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

        ax.set_title(CATEGORIES[cat]["name"], fontsize=11, fontweight='bold')
        ax.set_xlabel('Test SNR')
        ax.set_ylabel('Train SNR')

    plt.suptitle('Cross-Domain Noise Experiment Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison heatmap: {filepath}")


def save_baseline_degradation_plot(results: dict, filename: str):
    """Save a plot showing baseline degradation vs trained models."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ideal_acc = results.get("ideal", {}).get("clean", None)
    if ideal_acc:
        ideal_acc *= 100

    # Plot for Gaussian noise
    ax = axes[0]
    x = np.arange(len(SNR_LEVELS))
    width = 0.35

    # Baseline degradation (Clean→Gaussian)
    baseline_g = [results.get("baseline_gaussian", {}).get(snr, np.nan) * 100
                  for snr in SNR_LEVELS]

    # Diagonal of GG (Train Gaussian X → Test Gaussian X)
    gg_diagonal = []
    for snr in SNR_LEVELS:
        val = results.get("GG", {}).get((snr, snr), np.nan)
        gg_diagonal.append(val * 100 if not np.isnan(val) else np.nan)

    bars1 = ax.bar(x - width/2, baseline_g, width, label='Baseline (Clean Train)',
                   color='lightcoral', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, gg_diagonal, width, label='Matched Train (G→G diagonal)',
                   color='seagreen', edgecolor='black', alpha=0.8)

    if ideal_acc:
        ax.axhline(y=ideal_acc, color='gold', linestyle='--', linewidth=2,
                   label=f'Ideal (Clean→Clean): {ideal_acc:.1f}%')

    ax.set_xlabel('Test SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Gaussian Noise: Baseline vs Matched Training', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{snr}dB' for snr in SNR_LEVELS])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    # Plot for AudioSet noise
    ax = axes[1]

    # Baseline degradation (Clean→AudioSet)
    baseline_a = [results.get("baseline_audioset", {}).get(snr, np.nan) * 100
                  for snr in SNR_LEVELS]

    # Diagonal of AA
    aa_diagonal = []
    for snr in SNR_LEVELS:
        val = results.get("AA", {}).get((snr, snr), np.nan)
        aa_diagonal.append(val * 100 if not np.isnan(val) else np.nan)

    bars1 = ax.bar(x - width/2, baseline_a, width, label='Baseline (Clean Train)',
                   color='lightcoral', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, aa_diagonal, width, label='Matched Train (A→A diagonal)',
                   color='steelblue', edgecolor='black', alpha=0.8)

    if ideal_acc:
        ax.axhline(y=ideal_acc, color='gold', linestyle='--', linewidth=2,
                   label=f'Ideal (Clean→Clean): {ideal_acc:.1f}%')

    ax.set_xlabel('Test SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('AudioSet Noise: Baseline vs Matched Training', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{snr}dB' for snr in SNR_LEVELS])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Degradation: Clean Train vs Noise-Matched Train',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved baseline degradation plot: {filepath}")


def save_cross_domain_comparison(results: dict, filename: str):
    """Compare same-domain vs cross-domain training."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Get baseline values for reference
    ideal_acc = results.get("ideal", {}).get("clean", None)
    if ideal_acc:
        ideal_acc *= 100

    comparisons = [
        # (title, category1, category2, baseline_cat, ax_idx)
        ("Train Gaussian: Test on Gaussian vs AudioSet", "GG", "GA", "baseline_gaussian", (0, 0)),
        ("Train AudioSet: Test on AudioSet vs Gaussian", "AA", "AG", "baseline_audioset", (0, 1)),
        ("Test Gaussian: Trained on Gaussian vs AudioSet", "GG", "AG", "baseline_gaussian", (1, 0)),
        ("Test AudioSet: Trained on AudioSet vs Gaussian", "AA", "GA", "baseline_audioset", (1, 1)),
    ]

    for title, cat1, cat2, baseline_cat, (row, col) in comparisons:
        ax = axes[row, col]
        x = np.arange(len(SNR_LEVELS))
        width = 0.25

        # For first two plots: compare test performance at same test SNR
        if row == 0:
            # Same train SNR, different test types
            # Get diagonal values (matched train/test SNR)
            cat1_vals = []
            cat2_vals = []
            baseline_vals = []
            for snr in SNR_LEVELS:
                v1 = results.get(cat1, {}).get((snr, snr), np.nan)
                v2 = results.get(cat2, {}).get((snr, snr), np.nan)
                b = results.get(baseline_cat, {}).get(snr, np.nan)
                cat1_vals.append(v1 * 100 if not np.isnan(v1) else np.nan)
                cat2_vals.append(v2 * 100 if not np.isnan(v2) else np.nan)
                baseline_vals.append(b * 100 if not np.isnan(b) else np.nan)

            bars0 = ax.bar(x - width, baseline_vals, width, label='Baseline (Clean Train)',
                          color='lightcoral', edgecolor='black', alpha=0.8)
            bars1 = ax.bar(x, cat1_vals, width, label=CATEGORIES[cat1]["name"].split('(')[0],
                          color='seagreen', edgecolor='black', alpha=0.8)
            bars2 = ax.bar(x + width, cat2_vals, width, label=CATEGORIES[cat2]["name"].split('(')[0],
                          color='steelblue', edgecolor='black', alpha=0.8)
        else:
            # Same test type, different train types
            # Get values for same test SNR from different training
            cat1_vals = []
            cat2_vals = []
            baseline_vals = []
            for snr in SNR_LEVELS:
                v1 = results.get(cat1, {}).get((snr, snr), np.nan)
                v2 = results.get(cat2, {}).get((snr, snr), np.nan)
                b = results.get(baseline_cat, {}).get(snr, np.nan)
                cat1_vals.append(v1 * 100 if not np.isnan(v1) else np.nan)
                cat2_vals.append(v2 * 100 if not np.isnan(v2) else np.nan)
                baseline_vals.append(b * 100 if not np.isnan(b) else np.nan)

            bars0 = ax.bar(x - width, baseline_vals, width, label='Baseline (Clean Train)',
                          color='lightcoral', edgecolor='black', alpha=0.8)
            bars1 = ax.bar(x, cat1_vals, width,
                          label=f'Train {cat1[0]}' if len(cat1) == 2 else CATEGORIES[cat1]["name"],
                          color='seagreen', edgecolor='black', alpha=0.8)
            bars2 = ax.bar(x + width, cat2_vals, width,
                          label=f'Train {cat2[0]}' if len(cat2) == 2 else CATEGORIES[cat2]["name"],
                          color='steelblue', edgecolor='black', alpha=0.8)

        if ideal_acc:
            ax.axhline(y=ideal_acc, color='gold', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{snr}dB' for snr in SNR_LEVELS])
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Domain Noise Training Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cross-domain comparison: {filepath}")


def save_robustness_plot(results: dict, filename: str):
    """Save a bar plot for robustness experiments (train noisy → test clean)."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(SNR_LEVELS))
    width = 0.35

    # Gaussian robustness
    robust_g = [results.get("robust_gaussian", {}).get(snr, np.nan) * 100
                for snr in SNR_LEVELS]

    # AudioSet robustness
    robust_a = [results.get("robust_audioset", {}).get(snr, np.nan) * 100
                for snr in SNR_LEVELS]

    bars1 = ax.bar(x - width/2, robust_g, width, label='Train Gaussian → Test Clean',
                   color='seagreen', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, robust_a, width, label='Train AudioSet → Test Clean',
                   color='steelblue', edgecolor='black', alpha=0.8)

    # Ideal line
    ideal_acc = results.get("ideal", {}).get("clean", None)
    if ideal_acc:
        ax.axhline(y=ideal_acc * 100, color='gold', linestyle='--', linewidth=2,
                   label=f'Ideal (Clean→Clean): {ideal_acc*100:.1f}%')

    ax.set_xlabel('Training SNR (dB)')
    ax.set_ylabel('Accuracy on Clean Test (%)')
    ax.set_title('Robustness: Train on Noisy Data → Test on Clean Data',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{snr}dB' for snr in SNR_LEVELS])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved robustness plot: {filepath}")


def save_improvement_over_baseline(results: dict, filename: str):
    """Show improvement of noise training over clean baseline."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gaussian
    ax = axes[0]
    improvements = []
    snr_labels = []
    for snr in SNR_LEVELS:
        baseline = results.get("baseline_gaussian", {}).get(snr, np.nan)
        trained = results.get("GG", {}).get((snr, snr), np.nan)
        if not np.isnan(baseline) and not np.isnan(trained):
            imp = (trained - baseline) * 100
            improvements.append(imp)
            snr_labels.append(f'{snr}dB')

    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.bar(snr_labels, improvements, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Test SNR')
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('Gaussian: Improvement of Matched Training over Clean Baseline', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, improvements):
        ax.annotate(f'{val:+.1f}pp',
                   xy=(bar.get_x() + bar.get_width() / 2, val),
                   xytext=(0, 3 if val > 0 else -15), textcoords="offset points",
                   ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    # AudioSet
    ax = axes[1]
    improvements = []
    snr_labels = []
    for snr in SNR_LEVELS:
        baseline = results.get("baseline_audioset", {}).get(snr, np.nan)
        trained = results.get("AA", {}).get((snr, snr), np.nan)
        if not np.isnan(baseline) and not np.isnan(trained):
            imp = (trained - baseline) * 100
            improvements.append(imp)
            snr_labels.append(f'{snr}dB')

    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.bar(snr_labels, improvements, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Test SNR')
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('AudioSet: Improvement of Matched Training over Clean Baseline', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, improvements):
        ax.annotate(f'{val:+.1f}pp',
                   xy=(bar.get_x() + bar.get_width() / 2, val),
                   xytext=(0, 3 if val > 0 else -15), textcoords="offset points",
                   ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    plt.suptitle('Performance Gain from Noise-Matched Training', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved improvement plot: {filepath}")


def generate_summary_report(results: dict) -> str:
    """Generate a text summary report."""
    report = []
    report.append("=" * 80)
    report.append("CROSS-DOMAIN NOISE EXPERIMENTS - SUMMARY REPORT (v2)")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"SNR Levels: {SNR_LEVELS}")
    report.append("=" * 80)

    # Ideal (Clean→Clean)
    if "clean" in results.get("ideal", {}):
        report.append(f"\n*** IDEAL (Clean→Clean): {results['ideal']['clean']*100:.2f}% ***")
        report.append("(This is the upper bound - best possible performance)")

    # Baseline Degradation
    report.append(f"\n{'-'*60}")
    report.append("BASELINE DEGRADATION (Clean Train → Noisy Test)")
    report.append("Shows how clean-trained model breaks down with noise")
    report.append(f"{'-'*60}")

    for noise_type, cat in [("Gaussian", "baseline_gaussian"), ("AudioSet", "baseline_audioset")]:
        if results.get(cat):
            report.append(f"\n  {noise_type}:")
            for snr in SNR_LEVELS:
                if snr in results[cat]:
                    acc = results[cat][snr] * 100
                    ideal = results.get("ideal", {}).get("clean", 0) * 100
                    drop = ideal - acc if ideal else 0
                    report.append(f"    Test {snr}dB: {acc:.2f}% (drop: -{drop:.1f}pp)")

    # Matrix results
    for cat in ["GG", "AA", "GA", "AG"]:
        if results.get(cat):
            report.append(f"\n{'-'*60}")
            report.append(f"{CATEGORIES[cat]['name']}")
            report.append(f"{'-'*60}")

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
    report.append(f"\n{'-'*60}")
    report.append("ROBUSTNESS (Train Noisy → Test Clean)")
    report.append(f"{'-'*60}")
    for cat in ["robust_gaussian", "robust_audioset"]:
        if results.get(cat):
            report.append(f"\n  {CATEGORIES[cat]['name']}:")
            for snr in SNR_LEVELS:
                if snr in results[cat]:
                    report.append(f"    Train {snr}dB → Clean: {results[cat][snr]*100:.2f}%")

    report.append("\n" + "=" * 80)
    return "\n".join(report)


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Cross-Domain Noise Experiments - Results Analysis (v2)")
    print(f"SNR Levels: {SNR_LEVELS}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find experiments
    print("\nFinding experiment directories...")
    experiment_dirs = find_experiment_dirs()
    print(f"Found {len(experiment_dirs)} experiment directories")

    if not experiment_dirs:
        print("\nNo experiments found in results directory.")
        print("Run the training script first: sbatch scripts/train_cross_domain_v2.sh")
        return 1

    # Extract results
    print("\nExtracting results...")
    results = extract_results(experiment_dirs)

    # Print baseline info
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)

    if "clean" in results.get("ideal", {}):
        print(f"\nIdeal (Clean→Clean): {results['ideal']['clean']*100:.2f}%")

    print("\nBaseline Degradation (Clean Train → Noisy Test):")
    for noise_type, cat in [("Gaussian", "baseline_gaussian"), ("AudioSet", "baseline_audioset")]:
        if results.get(cat):
            for snr in SNR_LEVELS:
                if snr in results[cat]:
                    print(f"  {noise_type} @ {snr}dB: {results[cat][snr]*100:.2f}%")

    # Print and save matrix tables
    print("\n" + "=" * 60)
    print("RESULTS TABLES (4x4 matrices)")
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

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Baseline degradation plot
    save_baseline_degradation_plot(results, "baseline_degradation.png")

    # Improvement over baseline
    save_improvement_over_baseline(results, "improvement_over_baseline.png")

    # Comparison heatmaps
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

    # Cross-domain comparison
    save_cross_domain_comparison(results, "cross_domain_analysis.png")

    # Robustness plot
    if results.get("robust_gaussian") or results.get("robust_audioset"):
        save_robustness_plot(results, "robustness_barplot.png")

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
