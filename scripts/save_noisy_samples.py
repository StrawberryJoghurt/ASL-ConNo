#!/usr/bin/env python3
"""
Save audio samples with different noise levels for listening.
Uses real TIMIT audio files.
"""

import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("debug_audio_samples")
TIMIT_DIR = Path("autrainer-configurations/data/TIMIT-shared/default")
AUDIOSET_DIR = Path("autrainer-configurations/data/AudioSet-Balanced-Noise/default")


def add_gaussian_noise_waveform(waveform: torch.Tensor, snr_db: float, seed: int = 42) -> torch.Tensor:
    """Add Gaussian noise to waveform with target SNR."""
    torch.manual_seed(seed)

    # Calculate signal power
    signal_power = waveform.pow(2).mean()

    # Calculate noise power for target SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate noise with target power
    # For Gaussian: power = variance = std^2
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(waveform) * noise_std

    # Mix
    noisy = waveform + noise

    # Verify SNR
    actual_noise_power = noise.pow(2).mean()
    actual_snr = 10 * torch.log10(signal_power / actual_noise_power)
    print(f"    Target SNR: {snr_db} dB, Actual SNR: {actual_snr:.1f} dB")

    return noisy


def add_audioset_noise_waveform(waveform: torch.Tensor, snr_db: float,
                                 sample_rate: int = 16000, seed: int = 42) -> torch.Tensor:
    """Add AudioSet noise to waveform with target SNR."""
    np.random.seed(seed)

    # Find noise files
    noise_files = list(AUDIOSET_DIR.rglob("*.wav"))
    if not noise_files:
        print(f"No noise files found in {AUDIOSET_DIR}")
        return waveform

    # Load random noise
    noise_path = np.random.choice(noise_files)
    print(f"    Using noise: {noise_path.name}")
    noise_waveform, noise_sr = torchaudio.load(str(noise_path))

    # Resample if needed
    if noise_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(noise_sr, sample_rate)
        noise_waveform = resampler(noise_waveform)

    # Convert to mono
    if noise_waveform.shape[0] > 1:
        noise_waveform = noise_waveform.mean(dim=0, keepdim=True)

    # Match channels
    if waveform.shape[0] != noise_waveform.shape[0]:
        noise_waveform = noise_waveform.expand(waveform.shape[0], -1)

    # Match length
    target_length = waveform.shape[-1]
    noise_length = noise_waveform.shape[-1]

    if noise_length > target_length:
        start = np.random.randint(0, noise_length - target_length)
        noise_waveform = noise_waveform[:, start:start + target_length]
    elif noise_length < target_length:
        repeats = (target_length // noise_length) + 1
        noise_waveform = noise_waveform.repeat(1, repeats)[:, :target_length]

    # Calculate powers
    signal_power = waveform.pow(2).mean()
    noise_power_current = noise_waveform.pow(2).mean()

    # Target noise power
    noise_power_target = signal_power / (10 ** (snr_db / 10))

    # Scale noise (for amplitude: scale = sqrt(power_ratio))
    scale = torch.sqrt(noise_power_target / (noise_power_current + 1e-9))
    scaled_noise = noise_waveform * scale

    # Mix
    noisy = waveform + scaled_noise

    # Verify SNR
    actual_noise_power = scaled_noise.pow(2).mean()
    actual_snr = 10 * torch.log10(signal_power / actual_noise_power)
    print(f"    Target SNR: {snr_db} dB, Actual SNR: {actual_snr:.1f} dB")

    return noisy


def main():
    print("=" * 60)
    print("SAVING NOISY AUDIO SAMPLES (TIMIT)")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find a TIMIT wav file
    timit_files = list(TIMIT_DIR.rglob("*.WAV"))
    if not timit_files:
        print(f"No TIMIT files found in {TIMIT_DIR}")
        return 1

    # Find the longest file for better listening
    longest_duration = 0
    sample_path = timit_files[0]
    for f in timit_files:
        try:
            info = torchaudio.info(str(f))
            duration = info.num_frames / info.sample_rate
            if duration > longest_duration:
                longest_duration = duration
                sample_path = f
        except:
            pass
    print(f"\nUsing TIMIT file: {sample_path}")

    # Load audio
    waveform, sample_rate = torchaudio.load(str(sample_path))
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {waveform.shape[-1] / sample_rate:.2f}s")
    print(f"Shape: {waveform.shape}")
    print(f"Min: {waveform.min():.4f}, Max: {waveform.max():.4f}")

    # Save original
    original_path = OUTPUT_DIR / "00_original.wav"
    torchaudio.save(str(original_path), waveform, sample_rate)
    print(f"\nSaved: {original_path}")

    # SNR levels to test
    snr_levels = [20, 0, -20]

    # 1. Gaussian noise
    print("\n--- Gaussian Noise ---")
    for snr in snr_levels:
        print(f"  SNR = {snr} dB:")
        noisy = add_gaussian_noise_waveform(waveform.clone(), snr)

        # Clip to prevent distortion (keep in [-1, 1])
        noisy = torch.clamp(noisy, -1.0, 1.0)

        filename = f"gaussian_snr{snr:+d}dB.wav"
        filepath = OUTPUT_DIR / filename
        torchaudio.save(str(filepath), noisy, sample_rate)
        print(f"    Saved: {filepath}")

    # 2. AudioSet noise
    print("\n--- AudioSet Noise ---")
    if AUDIOSET_DIR.exists():
        for snr in snr_levels:
            print(f"  SNR = {snr} dB:")
            noisy = add_audioset_noise_waveform(waveform.clone(), snr, sample_rate)

            # Clip to prevent distortion
            noisy = torch.clamp(noisy, -1.0, 1.0)

            filename = f"audioset_snr{snr:+d}dB.wav"
            filepath = OUTPUT_DIR / filename
            torchaudio.save(str(filepath), noisy, sample_rate)
            print(f"    Saved: {filepath}")
    else:
        print(f"AudioSet directory not found: {AUDIOSET_DIR}")

    print("\n" + "=" * 60)
    print(f"All samples saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    print("""
Files created:
- 00_original.wav         : Clean TIMIT speech
- gaussian_snr+20dB.wav   : Light Gaussian noise (signal 100x stronger)
- gaussian_snr+0dB.wav    : Equal Gaussian noise (signal = noise)
- gaussian_snr-20dB.wav   : Heavy Gaussian noise (noise 100x stronger!)
- audioset_snr+20dB.wav   : Light AudioSet noise
- audioset_snr+0dB.wav    : Equal AudioSet noise
- audioset_snr-20dB.wav   : Heavy AudioSet noise

At -20dB the original speech should be almost completely masked by noise!
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
