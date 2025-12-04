#!/bin/bash
#SBATCH --job-name=cross_domain_v2
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=72:00:00
#SBATCH --output=slurm_cross_domain_v2_%j.out
#SBATCH --error=slurm_cross_domain_v2_%j.err

# ============================================================
# Cross-Domain Noise Experiments - Version 2
# ============================================================
# New SNR levels: -5dB, 0dB, 10dB, 20dB (replaced -20dB with -5dB and added 10dB)
#
# Experiments:
# - 1 Baseline Clean (Clean train + Clean test) - existing
# - 8 Baseline Degradation (Clean train + Noisy test) - NEW!
#     Shows how model degrades when tested on noisy data
# - 16 Same-domain Gaussian (GG): Train/Test Gaussian @ 4x4 SNR levels
# - 16 Same-domain AudioSet (AA): Train/Test AudioSet @ 4x4 SNR levels
# - 16 Cross-domain Gaussian→AudioSet (GA): 4x4 SNR levels
# - 16 Cross-domain AudioSet→Gaussian (AG): 4x4 SNR levels
# - 8 Robustness (Train noisy → Test clean)
#
# Total: 81 experiments
# Already completed (0, 20 dB combinations): ~21 experiments
# New experiments needed: ~60 experiments
#
# Skip logic: Already completed experiments will be skipped automatically
# ============================================================

cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# New SNR levels
SNR_LEVELS=("-5" "0" "10" "20")
NOISE_TYPES=("Gaussian" "AudioSet")

# Check GPU availability
echo "=========================================="
echo "GPU Status:"
nvidia-smi
echo "=========================================="

START_TIME=$(date +%s)
EXPERIMENT_COUNT=0
SKIPPED_COUNT=0
TOTAL_EXPERIMENTS=81

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

log_skip() {
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    echo "[SKIP] $1 - already exists"
}

# Check if experiment results exist
check_exists() {
    local exp_id=$1
    if [ -d "results/${exp_id}" ]; then
        # Check if training completed (has metrics.csv)
        if find "results/${exp_id}" -name "metrics.csv" -type f | grep -q .; then
            return 0  # exists and completed
        fi
    fi
    return 1  # doesn't exist or incomplete
}

# Generate config file for an experiment
generate_config() {
    local config_name=$1
    local exp_id=$2
    local train_noise_type=$3   # Gaussian, AudioSet, or None
    local train_snr=$4          # SNR value or empty
    local test_noise_type=$5    # Gaussian, AudioSet, or None
    local test_snr=$6           # SNR value or empty

    local config_file="conf/${config_name}.yaml"

    # Skip if config already exists
    if [ -f "$config_file" ]; then
        echo "Config exists: $config_file"
        return
    fi

    # Build train augmentation
    local train_aug=""
    if [ "$train_noise_type" != "None" ]; then
        local train_noise_impl="Gaussian"
        if [ "$train_noise_type" == "AudioSet" ]; then
            train_noise_impl="AudioSet"
        fi
        train_aug="train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: ${train_noise_type}Noise(${train_snr}dB)
  pipeline:
  - SNR_noise:
      _target_: autrainer.augmentations.spectrogram_augmentations.SNR_noise
      snr: ${train_snr}.0
      noise_type: ${train_noise_impl}
      p: 1.0
      generator_seed: 0"
    fi

    # Build test augmentation
    local test_aug=""
    if [ "$test_noise_type" != "None" ]; then
        local test_noise_impl="StaticGaussian"
        if [ "$test_noise_type" == "AudioSet" ]; then
            test_noise_impl="StaticAudioSet"
        fi
        test_aug="test_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: ${test_noise_type}Noise(${test_snr}dB)
  pipeline:
  - SNR_noise:
      _target_: autrainer.augmentations.spectrogram_augmentations.SNR_noise
      snr: ${test_snr}.0
      noise_type: ${test_noise_impl}
      p: 1.0
      generator_seed: 0"
    fi

    # Write config
    cat > "$config_file" << EOF
defaults:
- _autrainer_
- _self_
device: cuda
results_dir: results
experiment_id: ${exp_id}
iterations: 15
hydra:
  sweeper:
    params:
      +seed: 0
      +batch_size: 64
      +learning_rate: 0.001
      dataset: TIMIT-sentencetype-16k
      model: Cnn10-32k-T
      optimizer: Adam
${train_aug}
${test_aug}
EOF

    echo "Generated: $config_file"
}

# Run experiment with skip check
run_experiment() {
    local config_name=$1
    local exp_id=$2
    local description=$3

    if check_exists "$exp_id"; then
        log_skip "$description ($exp_id)"
        return
    fi

    log_progress "$description"
    autrainer train -cn "${config_name}.yaml"
}

# ============================================================
# Generate all configs first
# ============================================================
echo "=========================================="
echo "Generating configuration files..."
echo "=========================================="

# 1. Baseline Clean - use existing config
# baseline_clean already exists, no need to regenerate

# 2. Baseline Degradation: Clean Train + Noisy Test
for noise_type in "${NOISE_TYPES[@]}"; do
    for snr in "${SNR_LEVELS[@]}"; do
        config_name="CleanTrain_${noise_type}Test_${snr}dB"
        exp_id="CleanTrain_${noise_type}${snr}"
        generate_config "$config_name" "$exp_id" "None" "" "$noise_type" "$snr"
    done
done

# 3. Same-domain experiments (GG, AA)
for noise_type in "${NOISE_TYPES[@]}"; do
    type_short="${noise_type:0:1}${noise_type:0:1}"  # GG or AA
    for train_snr in "${SNR_LEVELS[@]}"; do
        for test_snr in "${SNR_LEVELS[@]}"; do
            config_name="${type_short}_train${train_snr}dB_test${test_snr}dB"
            exp_id="${type_short}_train${train_snr}_test${test_snr}"
            generate_config "$config_name" "$exp_id" "$noise_type" "$train_snr" "$noise_type" "$test_snr"
        done
    done
done

# 4. Cross-domain experiments (GA, AG)
# GA: Gaussian train → AudioSet test
for train_snr in "${SNR_LEVELS[@]}"; do
    for test_snr in "${SNR_LEVELS[@]}"; do
        config_name="GA_train${train_snr}dB_test${test_snr}dB"
        exp_id="GA_train${train_snr}_test${test_snr}"
        generate_config "$config_name" "$exp_id" "Gaussian" "$train_snr" "AudioSet" "$test_snr"
    done
done

# AG: AudioSet train → Gaussian test
for train_snr in "${SNR_LEVELS[@]}"; do
    for test_snr in "${SNR_LEVELS[@]}"; do
        config_name="AG_train${train_snr}dB_test${test_snr}dB"
        exp_id="AG_train${train_snr}_test${test_snr}"
        generate_config "$config_name" "$exp_id" "AudioSet" "$train_snr" "Gaussian" "$test_snr"
    done
done

# 5. Robustness: Noisy Train → Clean Test
for noise_type in "${NOISE_TYPES[@]}"; do
    type_short="${noise_type:0:1}"  # G or A
    for snr in "${SNR_LEVELS[@]}"; do
        config_name="Robust_${noise_type}${snr}dB_Clean"
        exp_id="Robust_${type_short}${snr}_Clean"
        generate_config "$config_name" "$exp_id" "$noise_type" "$snr" "None" ""
    done
done

echo "=========================================="
echo "Config generation complete!"
echo "=========================================="

# ============================================================
# Run experiments
# ============================================================

# 1. Baseline Clean (use existing)
run_experiment "Baseline_Clean" "baseline_clean" "Baseline: Clean Train + Clean Test"

# 2. Baseline Degradation (NEW!)
echo ""
echo "=========================================="
echo "BASELINE DEGRADATION: Clean Train + Noisy Test"
echo "=========================================="
for noise_type in "${NOISE_TYPES[@]}"; do
    for snr in "${SNR_LEVELS[@]}"; do
        config_name="CleanTrain_${noise_type}Test_${snr}dB"
        exp_id="CleanTrain_${noise_type}${snr}"
        run_experiment "$config_name" "$exp_id" "Clean Train → ${noise_type} Test @ ${snr}dB"
    done
done

# 3. Same-domain Gaussian (GG)
echo ""
echo "=========================================="
echo "SAME-DOMAIN GAUSSIAN (GG)"
echo "=========================================="
for train_snr in "${SNR_LEVELS[@]}"; do
    for test_snr in "${SNR_LEVELS[@]}"; do
        config_name="GG_train${train_snr}dB_test${test_snr}dB"
        exp_id="GG_train${train_snr}_test${test_snr}"
        run_experiment "$config_name" "$exp_id" "GG: Train ${train_snr}dB → Test ${test_snr}dB"
    done
done

# 4. Same-domain AudioSet (AA)
echo ""
echo "=========================================="
echo "SAME-DOMAIN AUDIOSET (AA)"
echo "=========================================="
for train_snr in "${SNR_LEVELS[@]}"; do
    for test_snr in "${SNR_LEVELS[@]}"; do
        config_name="AA_train${train_snr}dB_test${test_snr}dB"
        exp_id="AA_train${train_snr}_test${test_snr}"
        run_experiment "$config_name" "$exp_id" "AA: Train ${train_snr}dB → Test ${test_snr}dB"
    done
done

# 5. Cross-domain Gaussian→AudioSet (GA)
echo ""
echo "=========================================="
echo "CROSS-DOMAIN GAUSSIAN → AUDIOSET (GA)"
echo "=========================================="
for train_snr in "${SNR_LEVELS[@]}"; do
    for test_snr in "${SNR_LEVELS[@]}"; do
        config_name="GA_train${train_snr}dB_test${test_snr}dB"
        exp_id="GA_train${train_snr}_test${test_snr}"
        run_experiment "$config_name" "$exp_id" "GA: Gaussian ${train_snr}dB → AudioSet ${test_snr}dB"
    done
done

# 6. Cross-domain AudioSet→Gaussian (AG)
echo ""
echo "=========================================="
echo "CROSS-DOMAIN AUDIOSET → GAUSSIAN (AG)"
echo "=========================================="
for train_snr in "${SNR_LEVELS[@]}"; do
    for test_snr in "${SNR_LEVELS[@]}"; do
        config_name="AG_train${train_snr}dB_test${test_snr}dB"
        exp_id="AG_train${train_snr}_test${test_snr}"
        run_experiment "$config_name" "$exp_id" "AG: AudioSet ${train_snr}dB → Gaussian ${test_snr}dB"
    done
done

# 7. Robustness: Noisy Train → Clean Test
echo ""
echo "=========================================="
echo "ROBUSTNESS: Noisy Train → Clean Test"
echo "=========================================="
for noise_type in "${NOISE_TYPES[@]}"; do
    type_short="${noise_type:0:1}"
    for snr in "${SNR_LEVELS[@]}"; do
        config_name="Robust_${noise_type}${snr}dB_Clean"
        exp_id="Robust_${type_short}${snr}_Clean"
        run_experiment "$config_name" "$exp_id" "Robust: ${noise_type} ${snr}dB → Clean"
    done
done

# ============================================================
# COMPLETION
# ============================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Total experiments attempted: $TOTAL_EXPERIMENTS"
echo "Skipped (already completed): $SKIPPED_COUNT"
echo "Actually ran: $((EXPERIMENT_COUNT))"
echo "Total time: $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m $(($TOTAL_TIME % 60))s"
echo "=========================================="

# Show results locations
echo ""
echo "Results are in:"
ls -d results/*/ 2>/dev/null | wc -l
echo "experiment directories"
