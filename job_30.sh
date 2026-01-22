#!/usr/bin/env bash

#SBATCH --time=10:00:00
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000
#SBATCH --output=./autrain_job_30.out

# Change to your working directory
#cd /home/go35mig/zhiping/autrainer_example
cd /home/go35mig/zhiping/test/ASL-ConNo/

# Activate your virtual environment (NOT direnv in your case)
#source /home/go35mig/autrainenv/bin/activate
micromamba activate myenv

# Export your library path
#export LD_LIBRARY_PATH=/nix/store/7n3q3rgy5382di7ccrh3r6gk2xp51dh7-gcc-14.2.1.20250322-lib/lib:$LD_LIBRARY_PATH


echo "Job started on $(hostname)"
echo "Environment activated"
echo "which autrainer: $(which autrainer)"


which python
which autrainer
# Run your training command
#autrainer train -cn  GG_train30dB_test0dB.yaml device=cuda
#
#autrainer train -cn   GG_train30dB_test-5dB.yaml device=cuda
#
#autrainer train -cn   GG_train30dB_test10dB.yaml device=cuda
#
#autrainer train -cn   GG_train30dB_test20dB.yaml device=cuda

autrainer train -cn   GG_train30dB_test30dB.yaml device=cuda

#autrainer train -cn   AA_train30dB_test30dB.yaml device=cuda

#autrainer train -cn   AA_train30dB_test40dB.yaml device=cuda

echo "Job finished."
