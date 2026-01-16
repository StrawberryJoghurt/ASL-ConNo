#!/usr/bin/env bash

#SBATCH --time=12:00:00
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000
#SBATCH --output=./autrain_job_0.out

# Change to your working directory
cd /home/go35mig/zhiping/test/ASL-ConNo

# Activate your virtual environment
micromamba activate myenv

echo "Job started on $(hostname)"
echo "Environment activated"
which python
which aucurriculum

# ===========================================
# Curriculum Learning Experiment (15 epochs)
# ===========================================

# Step 1: Baseline training (generates checkpoints for scoring)
echo "=== Step 1: Baseline Training ==="
#aucurriculum train device=cuda
#aucurriculum train -cn curriculum_training hydra/sweeper=autrainer_filter_sweeper continue_training=false curriculum=None curriculum/sampling=None curriculum/scoring=None curriculum/pacing=None curriculum.pacing.initial_size=1 curriculum.pacing.final_iteration=0

## Step 2: Compute CumAcc difficulty scores
#echo "=== Step 2: CumAcc Scoring ==="
#aucurriculum curriculum device=cuda
aucurriculum train -cn curriculum_training hydra/sweeper=autrainer_filter_sweeper continue_training=false

## Step 3: Curriculum training (easy → hard)
#echo "=== Step 3: Curriculum Training ==="
#aucurriculum train -cn curriculum_training device=cuda
#
## Step 4: AntiCurriculum training (hard → easy)
#echo "=== Step 4: AntiCurriculum Training ==="
#aucurriculum train -cn anticurriculum_training device=cuda

echo "=== All steps completed ==="
echo "Job finished."
