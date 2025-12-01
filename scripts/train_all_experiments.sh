#!/bin/bash
#SBATCH --job-name=all_cross_domain_exp
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=20:00:00
#SBATCH --output=slurm_all_%j.out
#SBATCH --error=slurm_all_%j.err

# Activate virtual environment
cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Check GPU availability
nvidia-smi

echo "=========================================="
echo "Starting Full Experiment Suite"
echo "=========================================="

# 1. BASELINE: Clean training + Clean testing
echo "===== [1/7] Baseline: Clean Train + Clean Test ====="
autrainer train -cn Baseline_Clean.yaml

# 2-4. Train on Noisy, Test on Clean (data augmentation for robustness)
echo "===== [2/7] Train Noisy (-20dB) + Test Clean ====="
autrainer train -cn CrossDomainNoise_TrainNoisy_TestClean_neg20.yaml

echo "===== [3/7] Train Noisy (0dB) + Test Clean ====="
autrainer train -cn CrossDomainNoise_TrainNoisy_TestClean_0dB.yaml

echo "===== [4/7] Train Noisy (20dB) + Test Clean ====="
autrainer train -cn CrossDomainNoise_TrainNoisy_TestClean_20dB.yaml

# 5-7. Train on Noisy, Test on Noisy (matching conditions)
echo "===== [5/7] Train Noisy (-20dB) + Test Noisy (-20dB) ====="
autrainer train -cn CrossDomainNoise_neg20.yaml

echo "===== [6/7] Train Noisy (0dB) + Test Noisy (0dB) ====="
autrainer train -cn CrossDomainNoise_0dB.yaml

echo "===== [7/7] Train Noisy (20dB) + Test Noisy (20dB) ====="
autrainer train -cn CrossDomainNoise_20dB.yaml

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

# Show results locations
echo "Results are in:"
ls -lh results/*/training/
