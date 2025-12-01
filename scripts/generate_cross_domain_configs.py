#!/usr/bin/env python3
"""
Generate configuration files for comprehensive cross-domain noise experiments.

Experiment Matrix:
- SNR levels: -20dB, 0dB, 20dB
- Noise types: Gaussian (SNR_noise), AudioSet (CrossDomainNoise)

Experiments:
1. Baseline: Clean train + Clean test (1 exp)
2. Same-domain Gaussian (GG): 3x3 = 9 exp
3. Same-domain AudioSet (AA): 3x3 = 9 exp
4. Cross-domain Gaussian→AudioSet (GA): 3x3 = 9 exp
5. Cross-domain AudioSet→Gaussian (AG): 3x3 = 9 exp
6. Robustness (Train noisy → Test clean): 6 exp

Total: 43 experiments
"""

import os
import shutil
import yaml
from pathlib import Path

# Configuration
SNR_LEVELS = [-20, 0, 20]
OUTPUT_DIR = Path("conf")  # Put configs directly in conf/ (not subdirectory)
EXPERIMENT_BASE_ID = "cross_domain_full"

# Base configuration template
BASE_CONFIG = {
    "defaults": ["_autrainer_", "_self_"],
    "device": "cuda",
    "results_dir": "results",
    "iterations": 15,
    "hydra": {
        "sweeper": {
            "params": {
                "+seed": 0,
                "+batch_size": 64,
                "+learning_rate": 0.001,
                "dataset": "TIMIT-sentencetype-16k",
                "model": "Cnn10-32k-T",
                "optimizer": "Adam"
            }
        }
    }
}


def get_gaussian_augmentation(snr_db: int, is_train: bool = True) -> dict:
    """Generate Gaussian noise augmentation config with SNR."""
    noise_type = "Gaussian" if is_train else "StaticGaussian"
    return {
        "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
        "id": f"GaussianNoise({snr_db}dB)",
        "pipeline": [
            {
                "SNR_noise": {
                    "_target_": "autrainer.augmentations.spectrogram_augmentations.SNR_noise",
                    "snr": float(snr_db),
                    "noise_type": noise_type,
                    "p": 1.0,
                    "generator_seed": 0
                }
            }
        ]
    }


def get_audioset_augmentation(snr_db: int, is_train: bool = True) -> dict:
    """Generate AudioSet noise augmentation config with SNR."""
    csv_file = "train.csv" if is_train else "test.csv"
    return {
        "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
        "id": f"AudioSetNoise({snr_db}dB)",
        "pipeline": [
            {
                "CrossDomainNoise": {
                    "_target_": "autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise",
                    "noise_dir": "autrainer-configurations/data/AudioSet-Balanced-Noise",
                    "noise_csv": f"autrainer-configurations/data/AudioSet-Balanced-Noise/{csv_file}",
                    "snr_db": float(snr_db),
                    "sample_rate": 16000,
                    "n_fft": 512,
                    "hop_length": 160,
                    "n_mels": 64,
                    "noise_type": None,  # Use all noise types
                    "p": 1.0,
                    "generator_seed": 0
                }
            }
        ]
    }


def create_config(experiment_id: str, train_aug: dict = None, test_aug: dict = None) -> dict:
    """Create a complete configuration with optional augmentations."""
    config = BASE_CONFIG.copy()
    config = {
        "defaults": config["defaults"].copy(),
        "device": config["device"],
        "results_dir": config["results_dir"],
        "experiment_id": experiment_id,
        "iterations": config["iterations"],
        "hydra": {
            "sweeper": {
                "params": config["hydra"]["sweeper"]["params"].copy()
            }
        }
    }

    if train_aug is not None:
        config["train_augmentation"] = train_aug
    if test_aug is not None:
        config["test_augmentation"] = test_aug

    return config


def save_config(config: dict, filename: str):
    """Save configuration to YAML file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Created: {filepath}")
    return filepath


def generate_baseline():
    """Generate baseline (clean train + clean test) config."""
    config = create_config("baseline_clean")
    return save_config(config, "Baseline_Clean.yaml")


def generate_same_domain_gaussian():
    """Generate Same-domain Gaussian (GG) experiments: 3x3 matrix."""
    configs = []
    for train_snr in SNR_LEVELS:
        for test_snr in SNR_LEVELS:
            exp_id = f"GG_train{train_snr}_test{test_snr}"
            train_aug = get_gaussian_augmentation(train_snr, is_train=True)
            test_aug = get_gaussian_augmentation(test_snr, is_train=False)
            config = create_config(exp_id, train_aug, test_aug)
            filepath = save_config(config, f"GG_train{train_snr}dB_test{test_snr}dB.yaml")
            configs.append(filepath)
    return configs


def generate_same_domain_audioset():
    """Generate Same-domain AudioSet (AA) experiments: 3x3 matrix."""
    configs = []
    for train_snr in SNR_LEVELS:
        for test_snr in SNR_LEVELS:
            exp_id = f"AA_train{train_snr}_test{test_snr}"
            train_aug = get_audioset_augmentation(train_snr, is_train=True)
            test_aug = get_audioset_augmentation(test_snr, is_train=False)
            config = create_config(exp_id, train_aug, test_aug)
            filepath = save_config(config, f"AA_train{train_snr}dB_test{test_snr}dB.yaml")
            configs.append(filepath)
    return configs


def generate_cross_domain_gaussian_to_audioset():
    """Generate Cross-domain Gaussian→AudioSet (GA) experiments: 3x3 matrix."""
    configs = []
    for train_snr in SNR_LEVELS:
        for test_snr in SNR_LEVELS:
            exp_id = f"GA_train{train_snr}_test{test_snr}"
            train_aug = get_gaussian_augmentation(train_snr, is_train=True)
            test_aug = get_audioset_augmentation(test_snr, is_train=False)
            config = create_config(exp_id, train_aug, test_aug)
            filepath = save_config(config, f"GA_train{train_snr}dB_test{test_snr}dB.yaml")
            configs.append(filepath)
    return configs


def generate_cross_domain_audioset_to_gaussian():
    """Generate Cross-domain AudioSet→Gaussian (AG) experiments: 3x3 matrix."""
    configs = []
    for train_snr in SNR_LEVELS:
        for test_snr in SNR_LEVELS:
            exp_id = f"AG_train{train_snr}_test{test_snr}"
            train_aug = get_audioset_augmentation(train_snr, is_train=True)
            test_aug = get_gaussian_augmentation(test_snr, is_train=False)
            config = create_config(exp_id, train_aug, test_aug)
            filepath = save_config(config, f"AG_train{train_snr}dB_test{test_snr}dB.yaml")
            configs.append(filepath)
    return configs


def generate_robustness_experiments():
    """Generate Robustness experiments: Train noisy → Test clean."""
    configs = []

    # Gaussian train → Clean test
    for train_snr in SNR_LEVELS:
        exp_id = f"Robust_G{train_snr}_Clean"
        train_aug = get_gaussian_augmentation(train_snr, is_train=True)
        config = create_config(exp_id, train_aug, None)
        filepath = save_config(config, f"Robust_Gaussian{train_snr}dB_Clean.yaml")
        configs.append(filepath)

    # AudioSet train → Clean test
    for train_snr in SNR_LEVELS:
        exp_id = f"Robust_A{train_snr}_Clean"
        train_aug = get_audioset_augmentation(train_snr, is_train=True)
        config = create_config(exp_id, train_aug, None)
        filepath = save_config(config, f"Robust_AudioSet{train_snr}dB_Clean.yaml")
        configs.append(filepath)

    return configs


def main():
    """Generate all configuration files."""
    print("=" * 60)
    print("Generating Cross-Domain Noise Experiment Configurations")
    print("=" * 60)

    # Create output directory (conf/ should already exist with _autrainer_.yaml)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_configs = []

    # 1. Baseline
    print("\n[1/6] Generating Baseline config...")
    all_configs.append(generate_baseline())

    # 2. Same-domain Gaussian (GG)
    print("\n[2/6] Generating Same-domain Gaussian (GG) configs...")
    all_configs.extend(generate_same_domain_gaussian())

    # 3. Same-domain AudioSet (AA)
    print("\n[3/6] Generating Same-domain AudioSet (AA) configs...")
    all_configs.extend(generate_same_domain_audioset())

    # 4. Cross-domain Gaussian→AudioSet (GA)
    print("\n[4/6] Generating Cross-domain Gaussian→AudioSet (GA) configs...")
    all_configs.extend(generate_cross_domain_gaussian_to_audioset())

    # 5. Cross-domain AudioSet→Gaussian (AG)
    print("\n[5/6] Generating Cross-domain AudioSet→Gaussian (AG) configs...")
    all_configs.extend(generate_cross_domain_audioset_to_gaussian())

    # 6. Robustness experiments
    print("\n[6/6] Generating Robustness configs...")
    all_configs.extend(generate_robustness_experiments())

    print("\n" + "=" * 60)
    print(f"Total configurations generated: {len(all_configs)}")
    print("=" * 60)

    # Generate experiment list for bash script
    print("\nConfiguration files:")
    for config in all_configs:
        print(f"  - {config.name}")

    return all_configs


if __name__ == "__main__":
    main()
