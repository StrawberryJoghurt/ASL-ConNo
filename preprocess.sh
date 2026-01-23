#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-98         # Account/project
#SBATCH -p alvis                # Partition/queue
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -C MEM512
#SBATCH -t 0-10:00:00           # Walltime
#SBATCH --job-name=gnn
#SBATCH --output=train.log
#SBATCH --error=train.err

# Load CUDA module (MUST be before running apptainer)
module load CUDA/11.8.0

apptainer exec --nv \
    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:$LD_LIBRARY_PATH \
    /cephyr/users/zhiping/Alvis/ASL-ConNo/build/autrainer.sif \
    autrainer preprocess '++dataset.path=/mimer/NOBACKUP/groups/ulio_inverse/zhiping/data/SpeechCommands'
# Run autrainer
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/apps/Common/software/CUDA/11.8.0/lib64:$LD_LIBRARY_PATH \
    /cephyr/users/zhiping/Alvis/ASL-ConNo/build/autrainer.sif \
    autrainer train -cn AG_train-5dB_test-5dB.yaml device=cuda ++dataset.path=/mimer/NOBACKUP/groups/ulio_inverse/zhiping/data

