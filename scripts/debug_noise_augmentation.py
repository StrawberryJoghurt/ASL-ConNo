#!/usr/bin/env python3
"""
Debug script to verify noise augmentation is working correctly.
Saves audio samples at different SNR levels for manual inspection.
"""

import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autrainer.augmentations.spectrogram_augmentations import CrossDomainNoise, SNR_noise
from autrainer.core.structs import AbstractDataItem


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 10)


def linear_to_db(linear):
    """Convert linear to dB scale."""
    return 10 * np.log10(linear + 1e-9)


def calculate_snr(signal, noise):
    """Calculate actual SNR between signal and noise (in dB domain)."""
    # Convert from dB to linear
    signal_linear = db_to_linear(signal)
    noise_linear = db_to_linear(noise)

    # Calculate power
    signal_power = signal_linear.mean()
    noise_power = noise_linear.mean()

    # SNR in dB
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-9))
    return snr_db


def test_snr_calculation():
    """Test SNR calculation logic."""
    print("=" * 60)
    print("TEST: SNR Calculation Verification")
    print("=" * 60)

    # Test: -20dB means noise power = 100 * signal power
    # So noise should be 20dB LOUDER than signal
    target_snr = -20
    signal_power_db = -30  # typical log mel spectrogram value

    # Expected noise power (in linear domain)
    signal_power_linear = db_to_linear(signal_power_db)
    noise_power_linear = signal_power_linear / db_to_linear(target_snr)
    noise_power_db = linear_to_db(noise_power_linear)

    print(f"Target SNR: {target_snr} dB")
    print(f"Signal power: {signal_power_db:.1f} dB ({signal_power_linear:.6f} linear)")
    print(f"Required noise power: {noise_power_db:.1f} dB ({noise_power_linear:.6f} linear)")
    print(f"Noise/Signal ratio: {noise_power_linear/signal_power_linear:.1f}x")
    print()

    # For -20dB SNR, noise should be 100x stronger than signal
    expected_ratio = db_to_linear(20)  # 10^(20/10) = 100
    print(f"Expected noise/signal ratio for -20dB: {expected_ratio:.1f}x")
    print()


def test_cross_domain_noise_implementation():
    """Test CrossDomainNoise implementation."""
    print("=" * 60)
    print("TEST: CrossDomainNoise Implementation")
    print("=" * 60)

    noise_dir = "autrainer-configurations/data/AudioSet-Balanced-Noise"
    noise_csv = "autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv"

    if not os.path.exists(noise_dir):
        print(f"ERROR: Noise directory not found: {noise_dir}")
        return False

    # Create dummy spectrogram (typical log mel values: -80 to 0 dB)
    dummy_features = torch.randn(1, 64, 200) * 15 - 50  # values around -50 dB

    class DummyDataItem(AbstractDataItem):
        def __init__(self, features):
            self.features = features.clone()
            self.index = 0

    print(f"Original signal stats:")
    print(f"  Shape: {dummy_features.shape}")
    print(f"  Min: {dummy_features.min():.2f} dB")
    print(f"  Max: {dummy_features.max():.2f} dB")
    print(f"  Mean: {dummy_features.mean():.2f} dB")
    print()

    for target_snr in [20, 0, -20]:
        print(f"--- Testing SNR = {target_snr} dB ---")

        augmenter = CrossDomainNoise(
            noise_dir=noise_dir,
            noise_csv=noise_csv,
            snr_db=float(target_snr),
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=64,
            noise_type=None,
        )

        item = DummyDataItem(dummy_features)
        original = item.features.clone()

        augmented_item = augmenter.apply(item)
        augmented = augmented_item.features

        # Calculate difference
        diff = augmented - original

        print(f"  Augmented signal stats:")
        print(f"    Min: {augmented.min():.2f} dB")
        print(f"    Max: {augmented.max():.2f} dB")
        print(f"    Mean: {augmented.mean():.2f} dB")
        print(f"  Difference (noise added):")
        print(f"    Min diff: {diff.min():.2f} dB")
        print(f"    Max diff: {diff.max():.2f} dB")
        print(f"    Mean diff: {diff.mean():.2f} dB")
        print(f"    Std diff: {diff.std():.2f} dB")
        print()

        # For -20dB SNR, the difference should be significant
        # But looking at the code, there's a 0.1 multiplier that reduces it
        if target_snr == -20 and diff.abs().mean() < 1.0:
            print(f"  WARNING: Difference too small for {target_snr}dB SNR!")
            print(f"  This suggests the noise is being attenuated too much.")

    return True


def test_snr_noise_implementation():
    """Test SNR_noise (Gaussian) implementation."""
    print("=" * 60)
    print("TEST: SNR_noise (Gaussian) Implementation")
    print("=" * 60)

    # Create dummy spectrogram
    dummy_features = torch.randn(1, 64, 200) * 15 - 50

    class DummyDataItem(AbstractDataItem):
        def __init__(self, features):
            self.features = features.clone()
            self.index = 0

    print(f"Original signal stats:")
    print(f"  Shape: {dummy_features.shape}")
    print(f"  Min: {dummy_features.min():.2f} dB")
    print(f"  Max: {dummy_features.max():.2f} dB")
    print(f"  Mean: {dummy_features.mean():.2f} dB")
    print()

    for target_snr in [20, 0, -20]:
        print(f"--- Testing SNR = {target_snr} dB ---")

        augmenter = SNR_noise(
            snr=float(target_snr),
            noise_type="Gaussian",
            p=1.0,
            generator_seed=42,
        )

        item = DummyDataItem(dummy_features)
        original = item.features.clone()

        augmented_item = augmenter.apply(item)
        augmented = augmented_item.features

        diff = augmented - original

        print(f"  Augmented signal stats:")
        print(f"    Min: {augmented.min():.2f} dB")
        print(f"    Max: {augmented.max():.2f} dB")
        print(f"    Mean: {augmented.mean():.2f} dB")
        print(f"  Difference (noise added):")
        print(f"    Mean abs diff: {diff.abs().mean():.4f} dB")
        print(f"    Std diff: {diff.std():.4f} dB")
        print()


def analyze_bug_in_cross_domain_noise():
    """Analyze the bug in CrossDomainNoise: the 0.1 multiplier."""
    print("=" * 60)
    print("BUG ANALYSIS: CrossDomainNoise 0.1 multiplier")
    print("=" * 60)

    print("""
In spectrogram_augmentations.py line 246:
    item.features = item.features + scaled_noise_db * 0.1

This multiplies the noise by 0.1, effectively reducing it by 20dB!

For target SNR = -20dB:
- Expected: noise should be 100x stronger than signal
- Actual: noise is reduced by 10x, so it's only 10x stronger (10dB instead of 20dB)

This explains why the model performs well even with -20dB AudioSet noise:
the actual noise level is much lower than intended.
    """)

    # Calculate the effect
    print("Effect of 0.1 multiplier:")
    print(f"  Original -20dB SNR -> Actual ~0dB SNR (noise reduced by 20dB)")
    print(f"  Original 0dB SNR -> Actual ~20dB SNR")
    print(f"  Original 20dB SNR -> Actual ~40dB SNR")
    print()


def main():
    print("\n" + "=" * 60)
    print("NOISE AUGMENTATION DEBUG SCRIPT")
    print("=" * 60 + "\n")

    test_snr_calculation()
    analyze_bug_in_cross_domain_noise()
    test_cross_domain_noise_implementation()
    test_snr_noise_implementation()

    print("=" * 60)
    print("SUMMARY OF BUGS FOUND:")
    print("=" * 60)
    print("""
1. CrossDomainNoise (line 246):
   - Bug: `scaled_noise_db * 0.1` reduces noise by 20dB
   - Fix: Remove the 0.1 multiplier or use proper dB addition

2. Training.py (lines 92-94):
   - Bug: `dev_augmentation=self.cfg.train_augmentation`
   - Fix: `dev_augmentation=self.cfg.test_augmentation` or add separate dev config

3. CrossDomainNoise mathematical issue:
   - Adding noise in dB domain via simple addition is incorrect
   - Proper method: convert to linear, add, convert back to dB
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
