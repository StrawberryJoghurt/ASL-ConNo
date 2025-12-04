#!/bin/bash
#SBATCH --job-name=cross_domain_full
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=48:00:00
#SBATCH --output=slurm_cross_domain_%j.out
#SBATCH --error=slurm_cross_domain_%j.err

# ============================================================
# Cross-Domain Noise Experiments - Full Suite
# ============================================================
# 43 experiments total:
# - 1 Baseline (Clean train + Clean test)
# - 9 Same-domain Gaussian (GG): Train/Test Gaussian @ {-20, 0, 20}dB
# - 9 Same-domain AudioSet (AA): Train/Test AudioSet @ {-20, 0, 20}dB
# - 9 Cross-domain Gaussian→AudioSet (GA)
# - 9 Cross-domain AudioSet→Gaussian (AG)
# - 6 Robustness (Train noisy → Test clean)
# ============================================================

cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Check GPU availability
echo "=========================================="
echo "GPU Status:"
nvidia-smi
echo "=========================================="

START_TIME=$(date +%s)
EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=43

log_progress() {
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "[$EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS] $1"
    echo "Elapsed time: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m"
    echo "=========================================="
}

# ============================================================
# 1. BASELINE: Clean training + Clean testing
# ============================================================
log_progress "Baseline: Clean Train + Clean Test"
autrainer train -cn Baseline_Clean.yaml

# ============================================================
# 2. SAME-DOMAIN GAUSSIAN (GG): 3x3 matrix
# ============================================================
log_progress "GG: Train -20dB + Test -20dB"
autrainer train -cn GG_train-20dB_test-20dB.yaml

log_progress "GG: Train -20dB + Test 0dB"
autrainer train -cn GG_train-20dB_test0dB.yaml

log_progress "GG: Train -20dB + Test 20dB"
autrainer train -cn GG_train-20dB_test20dB.yaml

log_progress "GG: Train 0dB + Test -20dB"
autrainer train -cn GG_train0dB_test-20dB.yaml

log_progress "GG: Train 0dB + Test 0dB"
autrainer train -cn GG_train0dB_test0dB.yaml

log_progress "GG: Train 0dB + Test 20dB"
autrainer train -cn GG_train0dB_test20dB.yaml

log_progress "GG: Train 20dB + Test -20dB"
autrainer train -cn GG_train20dB_test-20dB.yaml

log_progress "GG: Train 20dB + Test 0dB"
autrainer train -cn GG_train20dB_test0dB.yaml

log_progress "GG: Train 20dB + Test 20dB"
autrainer train -cn GG_train20dB_test20dB.yaml

# ============================================================
# 3. SAME-DOMAIN AUDIOSET (AA): 3x3 matrix
# ============================================================
log_progress "AA: Train -20dB + Test -20dB"
autrainer train -cn AA_train-20dB_test-20dB.yaml

log_progress "AA: Train -20dB + Test 0dB"
autrainer train -cn AA_train-20dB_test0dB.yaml

log_progress "AA: Train -20dB + Test 20dB"
autrainer train -cn AA_train-20dB_test20dB.yaml

log_progress "AA: Train 0dB + Test -20dB"
autrainer train -cn AA_train0dB_test-20dB.yaml

log_progress "AA: Train 0dB + Test 0dB"
autrainer train -cn AA_train0dB_test0dB.yaml

log_progress "AA: Train 0dB + Test 20dB"
autrainer train -cn AA_train0dB_test20dB.yaml

log_progress "AA: Train 20dB + Test -20dB"
autrainer train -cn AA_train20dB_test-20dB.yaml

log_progress "AA: Train 20dB + Test 0dB"
autrainer train -cn AA_train20dB_test0dB.yaml

log_progress "AA: Train 20dB + Test 20dB"
autrainer train -cn AA_train20dB_test20dB.yaml

# ============================================================
# 4. CROSS-DOMAIN GAUSSIAN→AUDIOSET (GA): 3x3 matrix
# ============================================================
log_progress "GA: Train Gaussian -20dB + Test AudioSet -20dB"
autrainer train -cn GA_train-20dB_test-20dB.yaml

log_progress "GA: Train Gaussian -20dB + Test AudioSet 0dB"
autrainer train -cn GA_train-20dB_test0dB.yaml

log_progress "GA: Train Gaussian -20dB + Test AudioSet 20dB"
autrainer train -cn GA_train-20dB_test20dB.yaml

log_progress "GA: Train Gaussian 0dB + Test AudioSet -20dB"
autrainer train -cn GA_train0dB_test-20dB.yaml

log_progress "GA: Train Gaussian 0dB + Test AudioSet 0dB"
autrainer train -cn GA_train0dB_test0dB.yaml

log_progress "GA: Train Gaussian 0dB + Test AudioSet 20dB"
autrainer train -cn GA_train0dB_test20dB.yaml

log_progress "GA: Train Gaussian 20dB + Test AudioSet -20dB"
autrainer train -cn GA_train20dB_test-20dB.yaml

log_progress "GA: Train Gaussian 20dB + Test AudioSet 0dB"
autrainer train -cn GA_train20dB_test0dB.yaml

log_progress "GA: Train Gaussian 20dB + Test AudioSet 20dB"
autrainer train -cn GA_train20dB_test20dB.yaml

# ============================================================
# 5. CROSS-DOMAIN AUDIOSET→GAUSSIAN (AG): 3x3 matrix
# ============================================================
log_progress "AG: Train AudioSet -20dB + Test Gaussian -20dB"
autrainer train -cn AG_train-20dB_test-20dB.yaml

log_progress "AG: Train AudioSet -20dB + Test Gaussian 0dB"
autrainer train -cn AG_train-20dB_test0dB.yaml

log_progress "AG: Train AudioSet -20dB + Test Gaussian 20dB"
autrainer train -cn AG_train-20dB_test20dB.yaml

log_progress "AG: Train AudioSet 0dB + Test Gaussian -20dB"
autrainer train -cn AG_train0dB_test-20dB.yaml

log_progress "AG: Train AudioSet 0dB + Test Gaussian 0dB"
autrainer train -cn AG_train0dB_test0dB.yaml

log_progress "AG: Train AudioSet 0dB + Test Gaussian 20dB"
autrainer train -cn AG_train0dB_test20dB.yaml

log_progress "AG: Train AudioSet 20dB + Test Gaussian -20dB"
autrainer train -cn AG_train20dB_test-20dB.yaml

log_progress "AG: Train AudioSet 20dB + Test Gaussian 0dB"
autrainer train -cn AG_train20dB_test0dB.yaml

log_progress "AG: Train AudioSet 20dB + Test Gaussian 20dB"
autrainer train -cn AG_train20dB_test20dB.yaml

# ============================================================
# 6. ROBUSTNESS: Train noisy → Test clean
# ============================================================
log_progress "Robust: Train Gaussian -20dB + Test Clean"
autrainer train -cn Robust_Gaussian-20dB_Clean.yaml

log_progress "Robust: Train Gaussian 0dB + Test Clean"
autrainer train -cn Robust_Gaussian0dB_Clean.yaml

log_progress "Robust: Train Gaussian 20dB + Test Clean"
autrainer train -cn Robust_Gaussian20dB_Clean.yaml

log_progress "Robust: Train AudioSet -20dB + Test Clean"
autrainer train -cn Robust_AudioSet-20dB_Clean.yaml

log_progress "Robust: Train AudioSet 0dB + Test Clean"
autrainer train -cn Robust_AudioSet0dB_Clean.yaml

log_progress "Robust: Train AudioSet 20dB + Test Clean"
autrainer train -cn Robust_AudioSet20dB_Clean.yaml

# ============================================================
# COMPLETION
# ============================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total time: $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m $(($TOTAL_TIME % 60))s"
echo "=========================================="

# Show results locations
echo ""
echo "Results are in:"
ls -lh results/*/training/ 2>/dev/null || echo "No results found"
