#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-98         # Account/project
#SBATCH -p alvis                # Partition/queue
#SBATCH --cpus-per-task=8
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -C MEM512
#SBATCH -t 0-20:00:00           # Walltime
#SBATCH --job-name=gnn
#SBATCH --output=autrain.log
#SBATCH --error=autrain.err

# Load CUDA module (MUST be before running apptainer)
module load CUDA/11.8.0

module list 2>&1 || true
which python || true
which apptainer || true
nvcc --version || true

echo "===== GPU (HOST) ====="
nvidia-smi || true
nvidia-smi -L || true

# Run autrainer, BATCH -C MEM512
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:$LD_LIBRARY_PATH \
    /cephyr/users/zhiping/Alvis/ASL-ConNo/build/autrainer.sif \
    autrainer train -cn AG_train-5dB_test-5dB.yaml device=cuda ++dataset.path=/mimer/NOBACKUP/groups/ulio_inverse/zhiping/data/SpeechCommands

