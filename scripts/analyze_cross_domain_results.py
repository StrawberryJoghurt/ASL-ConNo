#!/usr/bin/env python3
"""
Analyze results from cross-domain noise experiments.
Generate NxN tables and visualizations for each experiment category.

Structure (v3):
- Ideal: Clean→Clean (upper bound, best possible performance)
- Baseline Degradation: Clean Train + Noisy Test (shows how model breaks)
- Same-domain: GG, AA, RR (train and test with same noise type)
- Cross-domain: GA, AG, GR, RG (train with one noise, test with another)
- Robustness: Train noisy → Test clean

Supports multiple noise types:
- G: Gaussian noise
- A: AudioSet noise  
- R: Rhythmic noise

SNR Levels: -5dB, 0dB, 10dB, 20dB, 30dB, 40dB
Handles incomplete experiments gracefully (shows N/A for missing data).

Usage:
    python analyze_cross_domain_results.py                         # Default: results/ -> analysis_results/
    python analyze_cross_domain_results.py -i results_rhythmic     # Custom input
    python analyze_cross_domain_results.py -i results_rhythmic -o analysis_results_rhythmic
"""

import os
import re
import sys
import json
import argparse
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

# Default directories (can be overridden via command line)
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("analysis_results")

# These will be set based on command line args
RESULTS_DIR = None
OUTPUT_DIR = None

# NEW: Updated SNR levels (excluding -20dB, added 30 and 40)
SNR_LEVELS = [-5, 0, 10, 20, 30, 40]

# Experiment categories - includes all possible noise types
# Categories will be filtered based on what's actually found in results
CATEGORIES = {
    "ideal": {
        "name": "Ideal (Clean→Clean)",
        "pattern": r"baseline_clean",
        "is_matrix": False
    },
    # Baseline degradation patterns
    "baseline_gaussian": {
        "name": "Baseline Degradation (Clean→Gaussian)",
        "pattern": r"(?:R_)?CleanTrain_Gaussian(-?\d+)",
        "is_matrix": False
    },
    "baseline_audioset": {
        "name": "Baseline Degradation (Clean→AudioSet)",
        "pattern": r"CleanTrain_AudioSet(-?\d+)",
        "is_matrix": False
    },
    "baseline_rhythmic": {
        "name": "Baseline Degradation (Clean→Rhythmic)",
        "pattern": r"(?:R_)?CleanTrain_Rhythmic(-?\d+)",
        "is_matrix": False
    },
    # Same-domain patterns
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
    "RR": {
        "name": "Same-Domain Rhythmic (R→R)",
        "pattern": r"RR_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    # Cross-domain patterns
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
    "GR": {
        "name": "Cross-Domain Gaussian→Rhythmic",
        "pattern": r"GR_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "RG": {
        "name": "Cross-Domain Rhythmic→Gaussian",
        "pattern": r"RG_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "AR": {
        "name": "Cross-Domain AudioSet→Rhythmic",
        "pattern": r"AR_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    "RA": {
        "name": "Cross-Domain Rhythmic→AudioSet",
        "pattern": r"RA_train(-?\d+)_test(-?\d+)",
        "is_matrix": True
    },
    # Robustness patterns
    "robust_gaussian": {
        "name": "Robustness (Gaussian→Clean)",
        "pattern": r"Robust_G(-?\d+)_Clean",
        "is_matrix": False
    },
    "robust_audioset": {
        "name": "Robustness (AudioSet→Clean)",
        "pattern": r"Robust_A(-?\d+)_Clean",
        "is_matrix": False
    },
    "robust_rhythmic": {
        "name": "Robustness (Rhythmic→Clean)",
        "pattern": r"Robust_R(-?\d+)_Clean",
        "is_matrix": False
    }
}


def safe_pct(value, default=np.nan):
    """Safely convert a value to percentage, handling None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value * 100


def safe_get_pct(results: dict, category: str, key, default=np.nan):
    """Safely get a value from results and convert to percentage."""
    value = results.get(category, {}).get(key, None)
    return safe_pct(value, default)


def find_run_directories() -> list:
    """Find all run directories (run_01, run_02, etc.) in results.
    
    If no run directories exist, returns [RESULTS_DIR] for backward compatibility.
    """
    run_dirs = []
    
    if not RESULTS_DIR.exists():
        return run_dirs
    
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("run_"):
            run_dirs.append(subdir)
    
    # Backward compatibility: if no run_XX directories, use RESULTS_DIR directly
    if not run_dirs:
        run_dirs = [RESULTS_DIR]
    
    return run_dirs


def find_experiment_dirs_in_run(run_dir: Path) -> dict:
    """Find all experiment directories in a single run directory.

    Returns dict mapping experiment_id (top-level dir name) to the actual
    experiment directory inside training/.
    """
    experiment_dirs = {}

    for results_subdir in run_dir.iterdir():
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


def find_experiment_dirs():
    """Find all experiment directories in results (backward compatible).
    
    Returns dict mapping experiment_id to experiment directory.
    Uses the first run directory if multiple exist.
    """
    run_dirs = find_run_directories()
    if not run_dirs:
        return {}
    
    # For single-run analysis, use first run
    return find_experiment_dirs_in_run(run_dirs[0])


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


def extract_all_runs_results() -> tuple:
    """Extract results from all run directories.
    
    Returns:
        - all_runs: list of (run_name, results_dict) tuples
        - aggregated: dict with mean values
        - std_dict: dict with std values
    """
    run_dirs = find_run_directories()
    all_runs = []
    
    for run_dir in run_dirs:
        run_name = run_dir.name if run_dir != RESULTS_DIR else "single_run"
        experiment_dirs = find_experiment_dirs_in_run(run_dir)
        if experiment_dirs:
            results = extract_results(experiment_dirs)
            all_runs.append((run_name, results))
    
    if not all_runs:
        return [], {cat: {} for cat in CATEGORIES}, {cat: {} for cat in CATEGORIES}
    
    # If only one run, return it without aggregation
    if len(all_runs) == 1:
        return all_runs, all_runs[0][1], {cat: {} for cat in CATEGORIES}
    
    # Aggregate across runs
    aggregated = {cat: {} for cat in CATEGORIES}
    std_dict = {cat: {} for cat in CATEGORIES}
    
    # Collect all keys from all runs for each category
    for cat in CATEGORIES:
        all_keys = set()
        for run_name, results in all_runs:
            all_keys.update(results.get(cat, {}).keys())
        
        for key in all_keys:
            values = []
            for run_name, results in all_runs:
                val = results.get(cat, {}).get(key)
                if val is not None:
                    values.append(val)
            
            if values:
                aggregated[cat][key] = np.mean(values)
                std_dict[cat][key] = np.std(values) if len(values) > 1 else 0.0
    
    return all_runs, aggregated, std_dict


def create_matrix_table(results: dict, category: str, std_results: dict = None) -> pd.DataFrame:
    """Create a NxN matrix table from results."""
    matrix = np.full((len(SNR_LEVELS), len(SNR_LEVELS)), np.nan)
    std_matrix = np.full((len(SNR_LEVELS), len(SNR_LEVELS)), np.nan) if std_results else None

    for (train_snr, test_snr), accuracy in results.get(category, {}).items():
        try:
            i = SNR_LEVELS.index(train_snr)
            j = SNR_LEVELS.index(test_snr)
            matrix[i, j] = accuracy
            if std_results and (train_snr, test_snr) in std_results.get(category, {}):
                std_matrix[i, j] = std_results[category][(train_snr, test_snr)]
        except ValueError:
            continue

    df = pd.DataFrame(
        matrix,
        index=[f"Train {snr}dB" for snr in SNR_LEVELS],
        columns=[f"Test {snr}dB" for snr in SNR_LEVELS]
    )
    
    if std_results:
        std_df = pd.DataFrame(
            std_matrix,
            index=[f"Train {snr}dB" for snr in SNR_LEVELS],
            columns=[f"Test {snr}dB" for snr in SNR_LEVELS]
        )
        return df, std_df
    
    return df


def print_matrix_table(df: pd.DataFrame, title: str, std_df: pd.DataFrame = None):
    """Print a formatted matrix table to console with optional std."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Format values as percentages with optional std
    formatted = pd.DataFrame(index=df.index, columns=df.columns, dtype=str)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if pd.notna(val):
                if std_df is not None and pd.notna(std_df.iloc[i, j]):
                    std_val = std_df.iloc[i, j]
                    formatted.iloc[i, j] = f"{val*100:.1f}±{std_val*100:.1f}"
                else:
                    formatted.iloc[i, j] = f"{val*100:.2f}%"
            else:
                formatted.iloc[i, j] = "N/A"

    print(formatted.to_string())


def save_heatmap(df: pd.DataFrame, title: str, filename: str, std_df: pd.DataFrame = None):
    """Save a heatmap visualization with optional std annotations."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert to percentage
    df_pct = df * 100
    std_pct = std_df * 100 if std_df is not None else None

    # Create annotation array with mean and ±std on separate lines
    annot_array = pd.DataFrame(index=df_pct.index, columns=df_pct.columns)
    for i in range(len(df_pct.index)):
        for j in range(len(df_pct.columns)):
            val = df_pct.iloc[i, j]
            if pd.notna(val):
                if std_pct is not None and pd.notna(std_pct.iloc[i, j]):
                    std_val = std_pct.iloc[i, j]
                    # Two lines: value on first, ±std on second
                    annot_array.iloc[i, j] = f"{val:.1f}\n±{std_val:.1f}"
                else:
                    annot_array.iloc[i, j] = f"{val:.1f}"
            else:
                annot_array.iloc[i, j] = "N/A"

    # Create mask for missing values
    mask = df_pct.isna()

    sns.heatmap(
        df_pct,
        annot=annot_array,
        fmt='',
        cmap='RdYlGn',
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
        ax=ax,
        mask=mask,
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )

    # Draw cells with N/A in gray
    if mask.any().any():
        for i in range(len(df_pct.index)):
            for j in range(len(df_pct.columns)):
                if mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightgray', alpha=0.5))
                    ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=12, color='gray')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Test SNR', fontsize=14)
    ax.set_ylabel('Train SNR', fontsize=14)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {filepath}")


def save_comparison_heatmap(results: dict, categories: list, filename: str, std_results: dict = None):
    """Save a comparison heatmap for multiple categories with optional std."""
    if not HAS_MATPLOTLIB:
        return

    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(7*n_cats, 8))

    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        if std_results and std_results.get(cat):
            df, std_df = create_matrix_table(results, cat, std_results)
            std_pct = std_df * 100
        else:
            df = create_matrix_table(results, cat)
            std_pct = None
        
        df_pct = df * 100

        # Create annotation array with mean±std on two lines
        annot_array = pd.DataFrame(index=df_pct.index, columns=df_pct.columns, dtype=str)
        for i in range(len(df_pct.index)):
            for j in range(len(df_pct.columns)):
                val = df_pct.iloc[i, j]
                if pd.notna(val):
                    if std_pct is not None and pd.notna(std_pct.iloc[i, j]):
                        std_val = std_pct.iloc[i, j]
                        annot_array.iloc[i, j] = f"{val:.1f}\n±{std_val:.1f}"
                    else:
                        annot_array.iloc[i, j] = f"{val:.1f}"
                else:
                    annot_array.iloc[i, j] = "N/A"
        
        mask = df_pct.isna()

        sns.heatmap(
            df_pct,
            annot=annot_array,
            fmt='',
            cmap='RdYlGn',
            center=50,
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
            mask=mask,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'}
        )

        # Draw N/A cells in gray
        if mask.any().any():
            for i in range(len(df_pct.index)):
                for j in range(len(df_pct.columns)):
                    if mask.iloc[i, j]:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightgray', alpha=0.5))
                        ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=10, color='gray')

        ax.set_title(CATEGORIES[cat]["name"], fontsize=13, fontweight='bold')
        ax.set_xlabel('Test SNR', fontsize=12)
        ax.set_ylabel('Train SNR', fontsize=12)
        ax.tick_params(labelsize=10)

    plt.suptitle('Cross-Domain Noise Experiment Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison heatmap: {filepath}")


def get_detected_noise_types(results: dict) -> list:
    """Detect which noise types are present in results."""
    noise_types = []
    
    # Check for Gaussian (G)
    if results.get("GG") or results.get("baseline_gaussian"):
        noise_types.append(("gaussian", "G", "Gaussian", "seagreen"))
    
    # Check for AudioSet (A)
    if results.get("AA") or results.get("baseline_audioset"):
        noise_types.append(("audioset", "A", "AudioSet", "steelblue"))
    
    # Check for Rhythmic (R)
    if results.get("RR") or results.get("baseline_rhythmic"):
        noise_types.append(("rhythmic", "R", "Rhythmic", "darkorange"))
    
    return noise_types


def get_ordered_matrix_cats(results: dict, noise_types: list) -> list:
    """Get matrix categories in the preferred order: same-domain first, then cross-domain.
    
    Order: GG, AA/RR (same-domain), then GA/GR, AG/RG (cross-domain)
    """
    same_domain = []
    cross_domain = []
    
    for noise_key1, code1, name1, _ in noise_types:
        for noise_key2, code2, name2, _ in noise_types:
            cat = f"{code1}{code2}"
            if results.get(cat):
                if code1 == code2:
                    same_domain.append(cat)
                else:
                    cross_domain.append(cat)
    
    # Return same-domain first, then cross-domain
    return same_domain + cross_domain


def save_baseline_degradation_plot(results: dict, filename: str, std_results: dict = None):
    """Save a plot showing baseline degradation vs trained models with optional error bars."""
    if not HAS_MATPLOTLIB:
        return

    noise_types = get_detected_noise_types(results)
    if not noise_types:
        print("  No noise types detected for baseline degradation plot")
        return
    
    n_plots = len(noise_types)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    ideal_acc = safe_get_pct(results, "ideal", "clean", None)
    x = np.arange(len(SNR_LEVELS))
    width = 0.35

    for idx, (noise_key, noise_code, noise_name, color) in enumerate(noise_types):
        ax = axes[idx]
        
        # Baseline degradation (Clean→Noise)
        baseline_cat = f"baseline_{noise_key}"
        baseline_vals = [safe_get_pct(results, baseline_cat, snr) for snr in SNR_LEVELS]
        
        # Diagonal of same-domain (Train Noise X → Test Noise X)
        same_domain_cat = f"{noise_code}{noise_code}"
        diagonal_vals = [safe_get_pct(results, same_domain_cat, (snr, snr)) for snr in SNR_LEVELS]
        
        # Get std values if available
        diagonal_std = None
        if std_results and std_results.get(same_domain_cat):
            diagonal_std = [safe_get_pct(std_results, same_domain_cat, (snr, snr), 0) for snr in SNR_LEVELS]

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Clean Train)',
                       color='lightcoral', edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, diagonal_vals, width, 
                       label=f'Matched Train ({noise_code}→{noise_code} diagonal)',
                       color=color, edgecolor='black', alpha=0.8,
                       yerr=diagonal_std if diagonal_std else None, capsize=3)

        if ideal_acc:
            ax.axhline(y=ideal_acc, color='gold', linestyle='--', linewidth=2,
                       label=f'Ideal (Clean→Clean): {ideal_acc:.1f}%')

        ax.set_xlabel('Test SNR (dB)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{noise_name} Noise: Baseline vs Matched Training', fontweight='bold')
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

    noise_types = get_detected_noise_types(results)
    if len(noise_types) < 2:
        print("  Need at least 2 noise types for cross-domain comparison")
        return
    
    # Build comparisons dynamically based on detected noise types
    noise_codes = [nt[1] for nt in noise_types]
    noise_names = {nt[1]: nt[2] for nt in noise_types}
    noise_colors = {nt[1]: nt[3] for nt in noise_types}
    baseline_cats = {nt[1]: f"baseline_{nt[0]}" for nt in noise_types}
    
    comparisons = []
    
    # For each pair of noise types, create comparison plots
    for i, code1 in enumerate(noise_codes):
        for code2 in noise_codes:
            if code1 != code2:
                # Same train, different test
                same_domain = f"{code1}{code1}"
                cross_domain = f"{code1}{code2}"
                if results.get(same_domain) and results.get(cross_domain):
                    comparisons.append((
                        f"Train {noise_names[code1]}: Test on {noise_names[code1]} vs {noise_names[code2]}",
                        same_domain, cross_domain, baseline_cats[code1],
                        noise_colors[code1], noise_colors[code2]
                    ))
    
    if not comparisons:
        print("  No cross-domain comparisons available")
        return
    
    # Limit to 4 plots max, arrange in grid
    comparisons = comparisons[:4]
    n_plots = len(comparisons)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    ideal_acc = safe_get_pct(results, "ideal", "clean", None)
    x = np.arange(len(SNR_LEVELS))
    width = 0.25

    for idx, (title, cat1, cat2, baseline_cat, color1, color2) in enumerate(comparisons):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        cat1_vals = [safe_get_pct(results, cat1, (snr, snr)) for snr in SNR_LEVELS]
        cat2_vals = [safe_get_pct(results, cat2, (snr, snr)) for snr in SNR_LEVELS]
        baseline_vals = [safe_get_pct(results, baseline_cat, snr) for snr in SNR_LEVELS]

        ax.bar(x - width, baseline_vals, width, label='Baseline (Clean Train)',
               color='lightcoral', edgecolor='black', alpha=0.8)
        ax.bar(x, cat1_vals, width, label=CATEGORIES[cat1]["name"].split('(')[0],
               color=color1, edgecolor='black', alpha=0.8)
        ax.bar(x + width, cat2_vals, width, label=CATEGORIES[cat2]["name"].split('(')[0],
               color=color2, edgecolor='black', alpha=0.8)

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

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Cross-Domain Noise Training Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cross-domain comparison: {filepath}")


def save_robustness_plot(results: dict, filename: str, std_results: dict = None):
    """Save a bar plot for robustness experiments (train noisy → test clean) with error bars."""
    if not HAS_MATPLOTLIB:
        return

    # Find which robustness results are available
    robust_types = []
    noise_info = [
        ("robust_gaussian", "Gaussian", "seagreen"),
        ("robust_audioset", "AudioSet", "steelblue"),
        ("robust_rhythmic", "Rhythmic", "darkorange"),
    ]
    
    for cat_key, name, color in noise_info:
        if results.get(cat_key):
            robust_types.append((cat_key, name, color))
    
    if not robust_types:
        print("  No robustness results available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(SNR_LEVELS))
    n_types = len(robust_types)
    width = 0.8 / n_types
    
    all_bars = []
    all_std = []
    for idx, (cat_key, name, color) in enumerate(robust_types):
        offset = (idx - (n_types - 1) / 2) * width
        values = [safe_get_pct(results, cat_key, snr) for snr in SNR_LEVELS]
        
        # Get std values if available
        std_vals = None
        if std_results and std_results.get(cat_key):
            std_vals = [safe_get_pct(std_results, cat_key, snr, 0) for snr in SNR_LEVELS]
        
        bars = ax.bar(x + offset, values, width, label=f'Train {name} → Test Clean',
                      color=color, edgecolor='black', alpha=0.8,
                      yerr=std_vals, capsize=3)
        all_bars.append(bars)
        all_std.append(std_vals)

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

    # Add value labels with std
    for bars, std_vals in zip(all_bars, all_std):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                std_text = f"\n±{std_vals[i]:.1f}" if std_vals and std_vals[i] > 0 else ""
                ax.annotate(f'{height:.1f}{std_text}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved robustness plot: {filepath}")


def save_improvement_over_baseline(results: dict, filename: str, std_results: dict = None):
    """Show improvement of noise training over clean baseline with optional error bars."""
    if not HAS_MATPLOTLIB:
        return

    noise_types = get_detected_noise_types(results)
    if not noise_types:
        print("  No noise types detected for improvement plot")
        return
    
    n_plots = len(noise_types)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for idx, (noise_key, noise_code, noise_name, color) in enumerate(noise_types):
        ax = axes[idx]
        improvements = []
        std_vals = []
        snr_labels = []
        
        baseline_cat = f"baseline_{noise_key}"
        same_domain_cat = f"{noise_code}{noise_code}"
        
        for snr in SNR_LEVELS:
            baseline = results.get(baseline_cat, {}).get(snr, np.nan)
            trained = results.get(same_domain_cat, {}).get((snr, snr), np.nan)
            if not np.isnan(baseline) and not np.isnan(trained):
                imp = (trained - baseline) * 100
                improvements.append(imp)
                snr_labels.append(f'{snr}dB')
                # Get std if available
                if std_results and std_results.get(same_domain_cat):
                    std = std_results.get(same_domain_cat, {}).get((snr, snr), 0) * 100
                    std_vals.append(std)
                else:
                    std_vals.append(0)

        if not improvements:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{noise_name}: No baseline comparison data', fontweight='bold')
            continue
            
        colors = ['green' if x > 0 else 'red' for x in improvements]
        yerr = std_vals if any(s > 0 for s in std_vals) else None
        bars = ax.bar(snr_labels, improvements, color=colors, edgecolor='black', alpha=0.8,
                     yerr=yerr, capsize=3)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Test SNR')
        ax.set_ylabel('Improvement (percentage points)')
        ax.set_title(f'{noise_name}: Improvement of Matched Training over Clean Baseline', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, improvements)):
            std_text = f"±{std_vals[i]:.1f}" if std_vals[i] > 0 else ""
            ax.annotate(f'{val:+.1f}pp{std_text}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3 if val > 0 else -15), textcoords="offset points",
                       ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    plt.suptitle('Performance Gain from Noise-Matched Training', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved improvement plot: {filepath}")


def generate_coverage_report(results: dict) -> str:
    """Generate a report showing which experiments are complete/missing."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("EXPERIMENT COVERAGE REPORT")
    report.append("=" * 80)

    total_expected = 0
    total_found = 0

    # Ideal
    report.append("\n📊 IDEAL (Clean→Clean)")
    if results.get("ideal", {}).get("clean"):
        report.append("  ✓ baseline_clean: COMPLETE")
        total_found += 1
    else:
        report.append("  ✗ baseline_clean: MISSING")
    total_expected += 1

    # Detect which noise types are present
    noise_types = get_detected_noise_types(results)
    
    # Baseline degradation - only show for detected noise types
    for noise_key, noise_code, noise_name, _ in noise_types:
        cat = f"baseline_{noise_key}"
        if results.get(cat):
            report.append(f"\n📊 BASELINE DEGRADATION (Clean→{noise_name})")
            for snr in SNR_LEVELS:
                total_expected += 1
                if results.get(cat, {}).get(snr) is not None:
                    report.append(f"  ✓ Test {snr}dB: COMPLETE")
                    total_found += 1
                else:
                    report.append(f"  ✗ Test {snr}dB: MISSING")

    # Matrix experiments - only show categories that have data (in preferred order)
    matrix_cats = get_ordered_matrix_cats(results, noise_types)
    
    for cat in matrix_cats:
        report.append(f"\n📊 {CATEGORIES[cat]['name']}")
        cat_expected = len(SNR_LEVELS) ** 2
        cat_found = len(results.get(cat, {}))
        missing = []
        for train_snr in SNR_LEVELS:
            for test_snr in SNR_LEVELS:
                total_expected += 1
                if results.get(cat, {}).get((train_snr, test_snr)) is not None:
                    total_found += 1
                else:
                    missing.append(f"train{train_snr}_test{test_snr}")

        if missing:
            report.append(f"  Found: {cat_found}/{cat_expected} experiments")
            report.append(f"  Missing: {', '.join(missing)}")
        else:
            report.append(f"  ✓ All {cat_expected} experiments COMPLETE")

    # Robustness - only show for detected noise types
    for noise_key, noise_code, noise_name, _ in noise_types:
        cat = f"robust_{noise_key}"
        if results.get(cat):
            report.append(f"\n📊 ROBUSTNESS ({noise_name}→Clean)")
            for snr in SNR_LEVELS:
                total_expected += 1
                if results.get(cat, {}).get(snr) is not None:
                    report.append(f"  ✓ Train {snr}dB: COMPLETE")
                    total_found += 1
                else:
                    report.append(f"  ✗ Train {snr}dB: MISSING")

    report.append(f"\n{'='*80}")
    if total_expected > 0:
        report.append(f"TOTAL COVERAGE: {total_found}/{total_expected} experiments ({100*total_found/total_expected:.1f}%)")
    else:
        report.append(f"TOTAL COVERAGE: {total_found} experiments found")
    report.append("=" * 80)

    return "\n".join(report)


def generate_summary_report(results: dict) -> str:
    """Generate a text summary report."""
    report = []
    report.append("=" * 80)
    report.append("CROSS-DOMAIN NOISE EXPERIMENTS - SUMMARY REPORT (v3)")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"SNR Levels: {SNR_LEVELS}")
    report.append(f"Results directory: {RESULTS_DIR}")
    report.append("=" * 80)

    # Detect noise types
    noise_types = get_detected_noise_types(results)
    noise_names = [nt[2] for nt in noise_types]
    report.append(f"Detected noise types: {', '.join(noise_names) if noise_names else 'None'}")

    # Ideal (Clean→Clean)
    if "clean" in results.get("ideal", {}):
        report.append(f"\n*** IDEAL (Clean→Clean): {results['ideal']['clean']*100:.2f}% ***")
        report.append("(This is the upper bound - best possible performance)")

    # Baseline Degradation
    baseline_found = any(results.get(f"baseline_{nt[0]}") for nt in noise_types)
    if baseline_found:
        report.append(f"\n{'-'*60}")
        report.append("BASELINE DEGRADATION (Clean Train → Noisy Test)")
        report.append("Shows how clean-trained model breaks down with noise")
        report.append(f"{'-'*60}")

        for noise_key, noise_code, noise_name, _ in noise_types:
            cat = f"baseline_{noise_key}"
            if results.get(cat):
                report.append(f"\n  {noise_name}:")
                for snr in SNR_LEVELS:
                    if snr in results[cat]:
                        acc = results[cat][snr] * 100
                        ideal = results.get("ideal", {}).get("clean", 0) * 100
                        drop = ideal - acc if ideal else 0
                        report.append(f"    Test {snr}dB: {acc:.2f}% (drop: -{drop:.1f}pp)")

    # Matrix results - dynamically find all matrix categories with data (in preferred order)
    matrix_cats = get_ordered_matrix_cats(results, noise_types)
    
    for cat in matrix_cats:
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
    robust_found = any(results.get(f"robust_{nt[0]}") for nt in noise_types)
    if robust_found:
        report.append(f"\n{'-'*60}")
        report.append("ROBUSTNESS (Train Noisy → Test Clean)")
        report.append(f"{'-'*60}")
        for noise_key, noise_code, noise_name, _ in noise_types:
            cat = f"robust_{noise_key}"
            if results.get(cat):
                report.append(f"\n  {CATEGORIES[cat]['name']}:")
                for snr in SNR_LEVELS:
                    if snr in results[cat]:
                        report.append(f"    Train {snr}dB → Clean: {results[cat][snr]*100:.2f}%")

    report.append("\n" + "=" * 80)
    return "\n".join(report)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze cross-domain noise experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze default results folder (uses first run if multiple exist)
    python analyze_cross_domain_results.py
    
    # Analyze rhythmic experiment results
    python analyze_cross_domain_results.py -i results_rhythmic -o analysis_results_rhythmic
    
    # Aggregate all runs and compute mean ± std
    python analyze_cross_domain_results.py -i results_rhythmic --aggregate
    
    # Use absolute paths
    python analyze_cross_domain_results.py -i /path/to/results -o /path/to/output
        """
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Input results directory (default: {DEFAULT_RESULTS_DIR})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for analysis results (default: analysis_results or analysis_<input_name>)"
    )
    parser.add_argument(
        "-a", "--aggregate",
        action="store_true",
        help="Aggregate results from all run directories (run_01, run_02, ...) and compute mean ± std"
    )
    parser.add_argument(
        "-r", "--run",
        type=str,
        default=None,
        help="Analyze a specific run only (e.g., run_01). If not specified, uses first run or aggregates all with --aggregate"
    )
    return parser.parse_args()


def main():
    """Main analysis function."""
    global RESULTS_DIR, OUTPUT_DIR
    
    # Parse command line arguments
    args = parse_args()
    
    # Set directories
    RESULTS_DIR = Path(args.input)
    
    # Handle specific run
    if args.run:
        specific_run = RESULTS_DIR / args.run
        if not specific_run.exists():
            print(f"Error: Run directory {specific_run} does not exist")
            return 1
        RESULTS_DIR = specific_run
    
    if args.output:
        OUTPUT_DIR = Path(args.output)
    else:
        # Auto-generate output dir name based on input
        if args.input == str(DEFAULT_RESULTS_DIR):
            OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        else:
            # e.g., results_rhythmic -> analysis_results_rhythmic
            input_name = Path(args.input).name
            if input_name.startswith("results"):
                suffix = input_name[7:]  # Remove "results" prefix
                OUTPUT_DIR = Path(f"analysis_results{suffix}")
            else:
                OUTPUT_DIR = Path(f"analysis_{input_name}")
        
        # Add suffix for aggregated results
        if args.aggregate:
            OUTPUT_DIR = Path(str(OUTPUT_DIR) + "_aggregated")
    
    print("=" * 60)
    print("Cross-Domain Noise Experiments - Results Analysis (v3)")
    print(f"SNR Levels: {SNR_LEVELS}")
    print(f"Input directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    if args.aggregate:
        print("Mode: AGGREGATING ALL RUNS (mean ± std)")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find runs and extract results
    std_results = None
    all_runs = None
    
    if args.aggregate:
        print("\nFinding all run directories...")
        all_runs, results, std_results = extract_all_runs_results()
        
        if not all_runs:
            print("\nNo experiments found in results directory.")
            print("Run the training script first.")
            return 1
        
        run_names = [r[0] for r in all_runs]
        print(f"Found {len(all_runs)} runs: {', '.join(run_names)}")
        
        total_exps = sum(len(find_experiment_dirs_in_run(RESULTS_DIR / r[0] if r[0] != "single_run" else RESULTS_DIR)) for r in all_runs)
        print(f"Total experiment directories across all runs: {total_exps}")
    else:
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

    # Detect noise types
    noise_types = get_detected_noise_types(results)
    
    # Print baseline degradation for all detected noise types
    baseline_found = False
    for noise_key, noise_code, noise_name, _ in noise_types:
        cat = f"baseline_{noise_key}"
        if results.get(cat):
            if not baseline_found:
                print("\nBaseline Degradation (Clean Train → Noisy Test):")
                baseline_found = True
            for snr in SNR_LEVELS:
                if snr in results[cat]:
                    print(f"  {noise_name} @ {snr}dB: {results[cat][snr]*100:.2f}%")

    # Find all matrix categories with data (in preferred order: same-domain first, then cross-domain)
    matrix_cats = get_ordered_matrix_cats(results, noise_types)

    # Print and save matrix tables
    if matrix_cats:
        print("\n" + "=" * 60)
        if args.aggregate and len(all_runs) > 1:
            print(f"RESULTS TABLES (NxN matrices) - Aggregated from {len(all_runs)} runs")
        else:
            print("RESULTS TABLES (NxN matrices)")
        print("=" * 60)

        for cat in matrix_cats:
            if results.get(cat):
                if std_results and std_results.get(cat):
                    df, std_df = create_matrix_table(results, cat, std_results)
                    title = CATEGORIES[cat]["name"]
                    if all_runs and len(all_runs) > 1:
                        title += f" (n={len(all_runs)} runs)"
                    print_matrix_table(df, title, std_df)
                    
                    # Save mean CSV
                    csv_path = OUTPUT_DIR / f"{cat}_matrix_mean.csv"
                    df.to_csv(csv_path)
                    print(f"  Saved mean CSV: {csv_path}")
                    
                    # Save std CSV
                    std_csv_path = OUTPUT_DIR / f"{cat}_matrix_std.csv"
                    std_df.to_csv(std_csv_path)
                    print(f"  Saved std CSV: {std_csv_path}")
                    
                    # Save heatmap with std
                    save_heatmap(df, title, f"{cat}_heatmap.png", std_df)
                else:
                    df = create_matrix_table(results, cat)
                    print_matrix_table(df, CATEGORIES[cat]["name"])
                    
                    # Save to CSV
                    csv_path = OUTPUT_DIR / f"{cat}_matrix.csv"
                    df.to_csv(csv_path)
                    print(f"  Saved CSV: {csv_path}")
                    
                    # Save heatmap without std
                    save_heatmap(df, CATEGORIES[cat]["name"], f"{cat}_heatmap.png")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Baseline degradation plot
    save_baseline_degradation_plot(results, "baseline_degradation.png", std_results)

    # Improvement over baseline
    save_improvement_over_baseline(results, "improvement_over_baseline.png", std_results)

    # Comparison heatmaps
    if len(matrix_cats) >= 2:
        save_comparison_heatmap(results, matrix_cats, "comparison_all.png", std_results)

        # Same-domain vs cross-domain
        same_domain = [c for c in matrix_cats if c[0] == c[1]]  # e.g., GG, AA, RR
        cross_domain = [c for c in matrix_cats if c[0] != c[1]]  # e.g., GA, AG, GR, RG

        if same_domain:
            save_comparison_heatmap(results, same_domain, "comparison_same_domain.png", std_results)
        if cross_domain:
            save_comparison_heatmap(results, cross_domain, "comparison_cross_domain.png", std_results)

    # Cross-domain comparison
    save_cross_domain_comparison(results, "cross_domain_analysis.png")

    # Robustness plot
    robust_found = any(results.get(f"robust_{nt[0]}") for nt in noise_types)
    if robust_found:
        save_robustness_plot(results, "robustness_barplot.png", std_results)

    # Generate coverage report
    print("\nGenerating coverage report...")
    coverage = generate_coverage_report(results)
    print(coverage)

    coverage_path = OUTPUT_DIR / "coverage_report.txt"
    with open(coverage_path, 'w') as f:
        f.write(coverage)
    print(f"\nSaved coverage report: {coverage_path}")

    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(results)
    print(report)

    # Save report (include coverage at top)
    report_path = OUTPUT_DIR / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write(coverage + "\n\n" + report)
    print(f"\nSaved report: {report_path}")

    # Save all results as JSON
    json_results = {
        "metadata": {
            "input_dir": str(RESULTS_DIR),
            "aggregated": args.aggregate,
            "num_runs": len(all_runs) if all_runs else 1,
            "runs": [r[0] for r in all_runs] if all_runs else ["single_run"],
        },
        "mean": {},
    }
    
    for cat, data in results.items():
        if data:
            json_results["mean"][cat] = {str(k): v for k, v in data.items()}
    
    # Add std if aggregated
    if std_results:
        json_results["std"] = {}
        for cat, data in std_results.items():
            if data:
                json_results["std"][cat] = {str(k): v for k, v in data.items()}
    
    # Save per-run results if aggregating
    if all_runs and len(all_runs) > 1:
        json_results["per_run"] = {}
        for run_name, run_results in all_runs:
            json_results["per_run"][run_name] = {}
            for cat, data in run_results.items():
                if data:
                    json_results["per_run"][run_name][cat] = {str(k): v for k, v in data.items()}

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
