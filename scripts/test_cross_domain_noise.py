#!/usr/bin/env python3
"""
Test script to verify CrossDomainNoise implementation.
Tests loading noise files and basic augmentation functionality.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add autrainer to path
sys.path.insert(0, str(Path(__file__).parent))

from autrainer.augmentations.spectrogram_augmentations import CrossDomainNoise
from autrainer.core.structs import AbstractDataItem


def test_noise_loading():
    """Test that noise files can be loaded."""
    print("=" * 60)
    print("TEST 1: Loading noise files")
    print("=" * 60)

    noise_dir = "autrainer-configurations/data/AudioSet-Balanced-Noise"
    noise_csv = "autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv"

    augmenter = CrossDomainNoise(
        noise_dir=noise_dir,
        noise_csv=noise_csv,
        snr_db=0.0,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=64,
        noise_type=None,
    )

    print(f"✓ Found {len(augmenter.noise_files)} noise files")

    # Test loading a specific noise file
    if len(augmenter.noise_files) > 0:
        test_noise = augmenter._load_noise_spectrogram(augmenter.noise_files[0])
        print(f"✓ Loaded noise spectrogram with shape: {test_noise.shape}")
    else:
        print("✗ No noise files found!")
        return False

    return True


def test_noise_filtering():
    """Test noise type filtering."""
    print("\n" + "=" * 60)
    print("TEST 2: Noise type filtering")
    print("=" * 60)

    noise_dir = "autrainer-configurations/data/AudioSet-Balanced-Noise"
    noise_csv = "autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv"

    for noise_type in ["Pink noise", "White noise", "Environmental noise", "Noise"]:
        augmenter = CrossDomainNoise(
            noise_dir=noise_dir,
            noise_csv=noise_csv,
            snr_db=0.0,
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=64,
            noise_type=noise_type,
        )
        print(f"✓ {noise_type}: {len(augmenter.noise_files)} files")

    return True


def test_augmentation():
    """Test applying augmentation to a sample."""
    print("\n" + "=" * 60)
    print("TEST 3: Applying augmentation")
    print("=" * 60)

    noise_dir = "autrainer-configurations/data/AudioSet-Balanced-Noise"
    noise_csv = "autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv"

    augmenter = CrossDomainNoise(
        noise_dir=noise_dir,
        noise_csv=noise_csv,
        snr_db=0.0,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=64,
        noise_type="Pink noise",
    )

    # Create a dummy data item (simulating log mel spectrogram)
    # Shape: (1, n_mels, time_steps)
    dummy_features = torch.randn(1, 64, 200) * 10 - 70  # Simulate dB values

    class DummyDataItem(AbstractDataItem):
        def __init__(self, features):
            self.features = features
            self.index = 0

    item = DummyDataItem(dummy_features)

    print(f"Original features shape: {item.features.shape}")
    print(f"Original features range: [{item.features.min():.2f}, {item.features.max():.2f}] dB")

    # Apply augmentation
    augmented_item = augmenter.apply(item)

    print(f"Augmented features shape: {augmented_item.features.shape}")
    print(f"Augmented features range: [{augmented_item.features.min():.2f}, {augmented_item.features.max():.2f}] dB")

    # Check that shape is preserved
    assert augmented_item.features.shape == dummy_features.shape, "Shape mismatch!"
    print("✓ Shape preserved")

    # Check that features changed
    diff = torch.abs(augmented_item.features - dummy_features).mean()
    print(f"Mean absolute difference: {diff:.4f} dB")

    if diff > 0.1:
        print("✓ Features were modified by augmentation")
    else:
        print("✗ Warning: Features barely changed!")

    return True


def test_length_matching():
    """Test length matching functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Length matching")
    print("=" * 60)

    noise_dir = "autrainer-configurations/data/AudioSet-Balanced-Noise"
    noise_csv = "autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv"

    augmenter = CrossDomainNoise(
        noise_dir=noise_dir,
        noise_csv=noise_csv,
        snr_db=0.0,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=64,
    )

    # Load a noise sample
    test_noise = augmenter._load_noise_spectrogram(augmenter.noise_files[0])
    print(f"Original noise shape: {test_noise.shape}")

    # Test shorter target
    short_target = 50
    matched_short = augmenter._match_length(test_noise, short_target)
    print(f"Matched to {short_target} frames: {matched_short.shape}")
    assert matched_short.shape[-1] == short_target, "Length mismatch!"

    # Test longer target
    long_target = 1000
    matched_long = augmenter._match_length(test_noise, long_target)
    print(f"Matched to {long_target} frames: {matched_long.shape}")
    assert matched_long.shape[-1] == long_target, "Length mismatch!"

    print("✓ Length matching works correctly")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING CrossDomainNoise Implementation")
    print("=" * 60 + "\n")

    tests = [
        ("Noise file loading", test_noise_loading),
        ("Noise type filtering", test_noise_filtering),
        ("Augmentation application", test_augmentation),
        ("Length matching", test_length_matching),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("Ready to run training experiments.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please fix issues before running training.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
