#!/usr/bin/env python3
"""
Download AudioSet rhythmic classes from HuggingFace using PyArrow directly.
Creates AudioSet-Balanced-Rhythmic with same structure as AudioSet-Balanced-Noise.
"""

import os
import csv
import random
import io
from pathlib import Path
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download, login
import pyarrow.parquet as pq

# HF token must be provided via environment for security.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Classes to download
TARGET_CLASSES = ["Bell", "Cymbal", "Drum", "Clapping"]

# Output directory
OUTPUT_DIR = Path("autrainer-configurations/data/AudioSet-Balanced-Rhythmic")
SAMPLES_PER_CLASS = 100
TRAIN_RATIO = 0.85
SEED = 42
TARGET_SR = 16000


def main():
    random.seed(SEED)

    # Login to HuggingFace
    print("Logging into HuggingFace...")
    if not HF_TOKEN:
        raise RuntimeError(
            "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running this script."
        )
    login(token=HF_TOKEN)

    # Create output directories
    for class_name in TARGET_CLASSES:
        (OUTPUT_DIR / "default" / class_name).mkdir(parents=True, exist_ok=True)
    
    all_train_entries = []
    all_test_entries = []
    
    # Download parquet files list
    print("Downloading AudioSet parquet files...")
    
    # Map split names to folder names
    split_folders = {'train': 'data/bal_train', 'test': 'data/eval'}
    
    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)
        
        # Get list of parquet files for this split
        from huggingface_hub import list_repo_files
        files = list_repo_files("agkphysics/AudioSet", repo_type="dataset")
        folder = split_folders[split]
        parquet_files = [f for f in files if f.startswith(folder) and f.endswith('.parquet')]
        
        print(f"Found {len(parquet_files)} parquet files for {split}")
        
        # Collect samples for each class
        class_samples = {c: [] for c in TARGET_CLASSES}
        
        for pf_idx, pf in enumerate(parquet_files):
            print(f"  Processing file {pf_idx+1}/{len(parquet_files)}: {pf}")
            
            # Download parquet file
            local_path = hf_hub_download(
                repo_id="agkphysics/AudioSet",
                filename=pf,
                repo_type="dataset"
            )
            
            # Read with pyarrow
            table = pq.read_table(local_path, columns=['human_labels', 'audio'])
            
            for i in range(len(table)):
                labels = table['human_labels'][i].as_py()
                
                # Check each target class
                for class_name in TARGET_CLASSES:
                    if any(class_name.lower() in str(l).lower() for l in labels):
                        # Get audio bytes
                        audio_data = table['audio'][i].as_py()
                        class_samples[class_name].append({
                            'audio': audio_data,
                            'labels': labels
                        })
            
            # Progress
            for c in TARGET_CLASSES:
                print(f"    {c}: {len(class_samples[c])} samples")
        
        # Save samples for each class
        for class_name in TARGET_CLASSES:
            samples = class_samples[class_name]
            random.shuffle(samples)
            
            n_samples = int(SAMPLES_PER_CLASS * (TRAIN_RATIO if split == 'train' else (1 - TRAIN_RATIO)))
            selected = samples[:n_samples]
            
            print(f"\n  Saving {len(selected)} {split} samples for {class_name}...")
            
            saved = 0
            for sample in selected:
                try:
                    audio_bytes = sample['audio']['bytes']
                    
                    # Decode audio
                    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                    
                    # Convert to mono if stereo
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)
                    
                    # Resample if needed
                    if sr != TARGET_SR:
                        import librosa
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                        sr = TARGET_SR
                    
                    # Save
                    prefix = "train" if split == 'train' else "test"
                    wav_path = OUTPUT_DIR / "default" / class_name / f"{prefix}_{saved:04d}.wav"
                    sf.write(str(wav_path), audio_array, sr)
                    
                    entries = all_train_entries if split == 'train' else all_test_entries
                    entries.append({
                        'path': f"default/{class_name}/{prefix}_{saved:04d}.wav",
                        'label': class_name
                    })
                    saved += 1
                    
                except Exception as e:
                    print(f"    Error: {e}")
            
            print(f"    Saved {saved} files")
    
    # Create CSVs
    print("\nCreating CSV files...")
    
    # Split train into train/dev
    random.shuffle(all_train_entries)
    split_idx = int(len(all_train_entries) * 0.85)
    train_data = all_train_entries[:split_idx]
    dev_data = all_train_entries[split_idx:]
    test_data = all_test_entries
    
    for csv_name, data in [('train.csv', train_data), ('dev.csv', dev_data), ('test.csv', test_data)]:
        csv_path = OUTPUT_DIR / csv_name
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'label'])
            writer.writeheader()
            writer.writerows(data)
        print(f"  {csv_name}: {len(data)} entries")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")


if __name__ == '__main__':
    main()
