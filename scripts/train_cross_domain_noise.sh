#!/bin/bash
#SBATCH --job-name=cross_domain_noise
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=10:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate virtual environment
cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Check GPU availability
nvidia-smi

# Print Python and PyTorch info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Train with CrossDomainNoise at -20dB
echo "===== Training with CrossDomainNoise at -20dB ====="
autrainer train -cn CrossDomainNoise_neg20.yaml

# Train with CrossDomainNoise at 0dB
echo "===== Training with CrossDomainNoise at 0dB ====="
autrainer train -cn CrossDomainNoise_0dB.yaml

# Train with CrossDomainNoise at 20dB
echo "===== Training with CrossDomainNoise at 20dB ====="
autrainer train -cn CrossDomainNoise_20dB.yaml

# Train with Pink Noise at 0dB
echo "===== Training with Pink Noise at 0dB ====="
autrainer train -cn CrossDomainNoise_PinkNoise_0dB.yaml

echo "===== All training jobs completed ====="
