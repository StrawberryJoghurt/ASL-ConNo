#!/usr/bin/env python3
"""
Verify that SNR augmentation produces correct SNR values.
This script measures the actual SNR after augmentation.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from autrainer.augmentations.spectrogram_augmentations import CrossDomainNoise, SNR_noise
from autrainer.core.structs import AbstractDataItem


class DummyDataItem(AbstractDataItem):
    def __init__(self, features):
        self.features = features.clone()
        self.index = 0


def measure_actual_snr(original_db, augmented_db):
    """
    Measure actual SNR between original signal and added noise.

    SNR = 10 * log10(P_signal / P_noise)

    Where:
    - P_signal = power of original signal
    - P_noise = power of (augmented - original) = power of noise added
    """
    # Convert to linear domain
    original_linear = 10 ** (original_db / 10)
    augmented_linear = 10 ** (augmented_db / 10)

    # Calculate noise (what was added)
    noise_linear = augmented_linear - original_linear

    # Calculate powers
    p_signal = original_linear.mean().item()
    p_noise = noise_linear.mean().item()

    # Calculate SNR
    if p_noise > 0:
        snr = 10 * np.log10(p_signal / p_noise)
    else:
        snr = float('inf')

    return snr, p_signal, p_noise


def test_cross_domain_noise():
    """Test CrossDomainNoise produces correct SNR."""
    print("=" * 60)
    print("TEST: CrossDomainNoise SNR Verification")
    print("=" * 60)

    noise_dir = "autrainer-configurations/data/AudioSet-Balanced-Noise"
    noise_csv = "autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv"

    if not os.path.exists(noise_dir):
        print(f"ERROR: Noise directory not found: {noise_dir}")
        return

    # Create realistic log mel spectrogram (values typically -80 to 0 dB)
    # Using values around -40 to -60 dB for typical speech
    torch.manual_seed(42)
    dummy_features = torch.randn(1, 64, 200) * 10 - 50  # Mean around -50 dB

    print(f"Original signal:")
    print(f"  Mean: {dummy_features.mean():.2f} dB")
    print(f"  Power (linear mean): {(10 ** (dummy_features / 10)).mean():.6e}")
    print()

    for target_snr in [20, 0, -20]:
        print(f"--- Target SNR = {target_snr} dB ---")

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

        actual_snr, p_signal, p_noise = measure_actual_snr(original, augmented)

        print(f"  P_signal: {p_signal:.6e}")
        print(f"  P_noise: {p_noise:.6e}")
        print(f"  Target SNR: {target_snr} dB")
        print(f"  Actual SNR: {actual_snr:.2f} dB")
        print(f"  Difference: {abs(actual_snr - target_snr):.2f} dB")

        if abs(actual_snr - target_snr) < 3:
            print(f"  ✓ SNR is close to target!")
        else:
            print(f"  ✗ SNR differs significantly from target")
        print()


def test_snr_noise():
    """Test SNR_noise (Gaussian) produces correct SNR."""
    print("=" * 60)
    print("TEST: SNR_noise (Gaussian) SNR Verification")
    print("=" * 60)

    torch.manual_seed(42)
    dummy_features = torch.randn(1, 64, 200) * 10 - 50

    print(f"Original signal:")
    print(f"  Mean: {dummy_features.mean():.2f} dB")
    print(f"  Power (linear mean): {(10 ** (dummy_features / 10)).mean():.6e}")
    print()

    for target_snr in [20, 0, -20]:
        print(f"--- Target SNR = {target_snr} dB ---")

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

        actual_snr, p_signal, p_noise = measure_actual_snr(original, augmented)

        print(f"  P_signal: {p_signal:.6e}")
        print(f"  P_noise: {p_noise:.6e}")
        print(f"  Target SNR: {target_snr} dB")
        print(f"  Actual SNR: {actual_snr:.2f} dB")
        print(f"  Difference: {abs(actual_snr - target_snr):.2f} dB")

        if abs(actual_snr - target_snr) < 3:
            print(f"  ✓ SNR is close to target!")
        else:
            print(f"  ✗ SNR differs significantly from target")
        print()


def explain_snr_math():
    """Explain SNR mathematics."""
    print("=" * 60)
    print("SNR MATHEMATICS EXPLANATION")
    print("=" * 60)
    print("""
SNR (Signal-to-Noise Ratio) Definition:
  SNR = 10 * log10(P_signal / P_noise)

For target SNR, we need:
  P_noise = P_signal / 10^(SNR/10)

Examples:
  SNR = 20 dB  →  P_noise = P_signal / 100   (noise is 100x weaker)
  SNR = 0 dB   →  P_noise = P_signal / 1     (noise equals signal)
  SNR = -20 dB →  P_noise = P_signal * 100   (noise is 100x stronger)

Important: dB values CANNOT be simply added!
  60 dB + 60 dB ≠ 120 dB

  Correct way:
    linear_a = 10^(60/10) = 1,000,000
    linear_b = 10^(60/10) = 1,000,000
    sum = 2,000,000
    result = 10*log10(2,000,000) = 63 dB

  So: 60 dB + 60 dB = 63 dB (only 3 dB increase!)
""")


def main():
    print("\n" + "=" * 60)
    print("SNR VERIFICATION SCRIPT")
    print("=" * 60 + "\n")

    explain_snr_math()
    test_cross_domain_noise()
    test_snr_noise()

    return 0


if __name__ == "__main__":
    sys.exit(main())
