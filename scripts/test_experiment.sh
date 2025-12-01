#!/bin/bash
#SBATCH --job-name=test_cross_domain
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:30:00
#SBATCH --output=slurm_test_%j.out
#SBATCH --error=slurm_test_%j.err

# Activate virtual environment
cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Check GPU availability
echo "===== GPU Info ====="
nvidia-smi

# Print Python and PyTorch info
echo "===== Python Info ====="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test with CrossDomainNoise at 0dB with just 1 iteration
echo "===== Testing CrossDomainNoise at 0dB (1 iteration) ====="
autrainer train -cn CrossDomainNoise_TrainNoisy_TestClean_0dB.yaml iterations=10

echo "===== Test completed ====="
