#!/usr/bin/env python3
"""
Generate configuration files for AudioSet-Balanced-Rhythmic experiments.
Creates the same configs as for AudioSet-Balanced-Noise but for the new rhythmic dataset.
"""

import os
from pathlib import Path
import yaml

# SNR levels to generate configs for
SNR_LEVELS = [-20, -10, 0, 10, 20]

# Dataset paths
RHYTHMIC_DATASET_DIR = "autrainer-configurations/data/AudioSet-Balanced-Rhythmic"

TEMPLATE = """defaults:
- _autrainer_
- _self_

device: cuda
results_dir: results
experiment_id: cross_domain_rhythmic
iterations: 15

# Test augmentation - apply rhythmic noise during testing
test_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: CrossDomainRhythmic({snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: {dataset_dir}
      noise_csv: {dataset_dir}/test.csv
      snr_db: {snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null  # Use all noise types (Bell, Cymbal, Drum, Clapping)
      p: 1.0
      generator_seed: 0

# Train augmentation - apply rhythmic noise during training
train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: CrossDomainRhythmic({snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: {dataset_dir}
      noise_csv: {dataset_dir}/train.csv
      snr_db: {snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null  # Use all noise types (Bell, Cymbal, Drum, Clapping)
      p: 1.0
      generator_seed: 0

hydra:
  sweeper:
    params:
      +seed: 0
      +batch_size: 64
      +learning_rate: 0.001
      dataset: TIMIT-sentencetype-16k
      model: Cnn10-32k-T
      optimizer: Adam
"""


def main():
    conf_dir = Path("conf")
    conf_dir.mkdir(exist_ok=True)
    
    print("Generating CrossDomainRhythmic configs...")
    
    for snr in SNR_LEVELS:
        # Handle negative SNR in filename
        if snr < 0:
            snr_str = f"neg{abs(snr)}"
        else:
            snr_str = str(snr)
        
        filename = f"CrossDomainRhythmic_{snr_str}dB.yaml"
        filepath = conf_dir / filename
        
        config = TEMPLATE.format(
            snr=snr,
            dataset_dir=RHYTHMIC_DATASET_DIR
        )
        
        with open(filepath, 'w') as f:
            f.write(config)
        
        print(f"  Created {filepath}")
    
    print(f"\nGenerated {len(SNR_LEVELS)} config files.")
    print("\nUsage examples:")
    print("  autrainer train -cn CrossDomainRhythmic_0dB")
    print("  autrainer train -cn CrossDomainRhythmic_neg20dB")
    print("  autrainer train -cn CrossDomainRhythmic_20dB")


if __name__ == "__main__":
    main()
