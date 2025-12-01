#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8000
#SBATCH --job-name=timit-train
#SBATCH --output=./slurm_%A.out
#SBATCH --error=./slurm_%A.err

# Print some info
echo "Starting get noise on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Navigate to project directory
cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo

# Activate virtual environment
source .venv/bin/activate

# Run training
python get_noise_sigma.py

# Print completion info
date
echo "Get noise completed"

