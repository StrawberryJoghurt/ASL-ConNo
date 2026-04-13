#!/bin/bash
#SBATCH --job-name=cross_domain_rhythmic
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=96:00:00
#SBATCH --output=slurm_cross_domain_rhythmic_%j.out
#SBATCH --error=slurm_cross_domain_rhythmic_%j.err

# ============================================================
# Cross-Domain Rhythmic Noise Experiments
# ============================================================
# Same as train_cross_domain_v3.sh but with AudioSet-Balanced-Rhythmic
# R = Rhythmic (Bell, Cymbal, Drum, Clapping)
# G = Gaussian
# ============================================================

cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Dataset paths
RHYTHMIC_DIR="autrainer-configurations/data/AudioSet-Balanced-Rhythmic"
RESULTS_DIR="results_rhythmic"

SNR_LEVELS=("-5" "0" "10" "20" "30" "40")

echo "=========================================="
echo "Cross-Domain Rhythmic Experiments"
echo "Dataset: $RHYTHMIC_DIR"
echo "Results: $RESULTS_DIR"
echo "SNR Levels: ${SNR_LEVELS[*]}"
nvidia-smi
echo "=========================================="

START_TIME=$(date +%s)
EXPERIMENT_COUNT=0
SKIPPED_COUNT=0

log_progress() {
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "[$EXPERIMENT_COUNT] $1"
    echo "Elapsed: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m"
    echo "=========================================="
}

log_skip() {
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    echo "[SKIP] $1"
}

check_exists() {
    local exp_id=$1
    if [ -d "${RESULTS_DIR}/${exp_id}" ]; then
        if find "${RESULTS_DIR}/${exp_id}" -name "test_holistic.yaml" -type f 2>/dev/null | grep -q .; then
            return 0
        fi
    fi
    return 1
}

# Generate Gaussian config
gen_gaussian() {
    local cfg=$1 exp=$2 train_snr=$3 test_snr=$4
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""
    if [ -n "$train_snr" ]; then
        train_aug="train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: GaussianNoise(${train_snr}dB)
  pipeline:
  - SNR_noise:
      _target_: autrainer.augmentations.spectrogram_augmentations.SNR_noise
      snr: ${train_snr}.0
      noise_type: Gaussian
      p: 1.0
      generator_seed: 0"
    fi
    if [ -n "$test_snr" ]; then
        test_aug="test_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: GaussianNoise(${test_snr}dB)
  pipeline:
  - SNR_noise:
      _target_: autrainer.augmentations.spectrogram_augmentations.SNR_noise
      snr: ${test_snr}.0
      noise_type: StaticGaussian
      p: 1.0
      generator_seed: 0"
    fi

    cat > "$f" << EOF
defaults:
- _autrainer_
- _self_
device: cuda
results_dir: ${RESULTS_DIR}
experiment_id: ${exp}
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
    echo "Generated: $f"
}

# Generate Rhythmic config (CrossDomainNoise with AudioSet-Balanced-Rhythmic)
gen_rhythmic() {
    local cfg=$1 exp=$2 train_snr=$3 test_snr=$4
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""
    if [ -n "$train_snr" ]; then
        train_aug="train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: RhythmicNoise(${train_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${RHYTHMIC_DIR}
      noise_csv: ${RHYTHMIC_DIR}/train.csv
      snr_db: ${train_snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null
      p: 1.0
      generator_seed: 0"
    fi
    if [ -n "$test_snr" ]; then
        test_aug="test_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: RhythmicNoise(${test_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${RHYTHMIC_DIR}
      noise_csv: ${RHYTHMIC_DIR}/test.csv
      snr_db: ${test_snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null
      p: 1.0
      generator_seed: 0"
    fi

    cat > "$f" << EOF
defaults:
- _autrainer_
- _self_
device: cuda
results_dir: ${RESULTS_DIR}
experiment_id: ${exp}
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
    echo "Generated: $f"
}

# Generate cross-domain config (Gaussian <-> Rhythmic)
gen_cross() {
    local cfg=$1 exp=$2 train_type=$3 train_snr=$4 test_type=$5 test_snr=$6
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""

    if [ "$train_type" == "G" ]; then
        train_aug="train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: GaussianNoise(${train_snr}dB)
  pipeline:
  - SNR_noise:
      _target_: autrainer.augmentations.spectrogram_augmentations.SNR_noise
      snr: ${train_snr}.0
      noise_type: Gaussian
      p: 1.0
      generator_seed: 0"
    else
        train_aug="train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: RhythmicNoise(${train_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${RHYTHMIC_DIR}
      noise_csv: ${RHYTHMIC_DIR}/train.csv
      snr_db: ${train_snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null
      p: 1.0
      generator_seed: 0"
    fi

    if [ "$test_type" == "G" ]; then
        test_aug="test_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: GaussianNoise(${test_snr}dB)
  pipeline:
  - SNR_noise:
      _target_: autrainer.augmentations.spectrogram_augmentations.SNR_noise
      snr: ${test_snr}.0
      noise_type: StaticGaussian
      p: 1.0
      generator_seed: 0"
    else
        test_aug="test_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: RhythmicNoise(${test_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${RHYTHMIC_DIR}
      noise_csv: ${RHYTHMIC_DIR}/test.csv
      snr_db: ${test_snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null
      p: 1.0
      generator_seed: 0"
    fi

    cat > "$f" << EOF
defaults:
- _autrainer_
- _self_
device: cuda
results_dir: ${RESULTS_DIR}
experiment_id: ${exp}
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
    echo "Generated: $f"
}

run_exp() {
    local cfg=$1 exp=$2 desc=$3
    check_exists "$exp" && { log_skip "$desc"; return; }
    log_progress "$desc"
    autrainer train -cn "${cfg}.yaml"
}

# ============================================================
echo "Generating configs..."
# ============================================================

# Baseline degradation (Clean train -> Rhythmic/Gaussian test)
for s in "${SNR_LEVELS[@]}"; do
    gen_gaussian "R_CleanTrain_GaussianTest_${s}dB" "R_CleanTrain_Gaussian${s}" "" "$s"
    gen_rhythmic "R_CleanTrain_RhythmicTest_${s}dB" "R_CleanTrain_Rhythmic${s}" "" "$s"
done

# GG (Gaussian train -> Gaussian test) - same as v3 but saved to results_rhythmic
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_gaussian "R_GG_train${t}dB_test${e}dB" "R_GG_train${t}_test${e}" "$t" "$e"
    done
done

# RR (Rhythmic train -> Rhythmic test)
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_rhythmic "R_RR_train${t}dB_test${e}dB" "R_RR_train${t}_test${e}" "$t" "$e"
    done
done

# GR (Gaussian train -> Rhythmic test)
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_cross "R_GR_train${t}dB_test${e}dB" "R_GR_train${t}_test${e}" "G" "$t" "R" "$e"
    done
done

# RG (Rhythmic train -> Gaussian test)
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_cross "R_RG_train${t}dB_test${e}dB" "R_RG_train${t}_test${e}" "R" "$t" "G" "$e"
    done
done

# Robustness (noisy train -> clean test)
for s in "${SNR_LEVELS[@]}"; do
    gen_gaussian "R_Robust_Gaussian${s}dB_Clean" "R_Robust_G${s}_Clean" "$s" ""
    gen_rhythmic "R_Robust_Rhythmic${s}dB_Clean" "R_Robust_R${s}_Clean" "$s" ""
done

echo "Config generation complete!"

# ============================================================
echo "Running experiments..."
# ============================================================

# Baseline
run_exp "Baseline_Clean" "R_baseline_clean" "Baseline Clean"

# Baseline degradation
for s in "${SNR_LEVELS[@]}"; do
    run_exp "R_CleanTrain_GaussianTest_${s}dB" "R_CleanTrain_Gaussian${s}" "Clean→Gaussian ${s}dB"
done
for s in "${SNR_LEVELS[@]}"; do
    run_exp "R_CleanTrain_RhythmicTest_${s}dB" "R_CleanTrain_Rhythmic${s}" "Clean→Rhythmic ${s}dB"
done

# GG
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "R_GG_train${t}dB_test${e}dB" "R_GG_train${t}_test${e}" "GG ${t}→${e}dB"
    done
done

# RR
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "R_RR_train${t}dB_test${e}dB" "R_RR_train${t}_test${e}" "RR ${t}→${e}dB"
    done
done

# GR
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "R_GR_train${t}dB_test${e}dB" "R_GR_train${t}_test${e}" "GR G${t}→R${e}dB"
    done
done

# RG
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "R_RG_train${t}dB_test${e}dB" "R_RG_train${t}_test${e}" "RG R${t}→G${e}dB"
    done
done

# Robustness
for s in "${SNR_LEVELS[@]}"; do
    run_exp "R_Robust_Gaussian${s}dB_Clean" "R_Robust_G${s}_Clean" "Robust G${s}→Clean"
done
for s in "${SNR_LEVELS[@]}"; do
    run_exp "R_Robust_Rhythmic${s}dB_Clean" "R_Robust_R${s}_Clean" "Robust R${s}→Clean"
done

# ============================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "COMPLETED!"
echo "Run: $EXPERIMENT_COUNT | Skipped: $SKIPPED_COUNT"
echo "Time: $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m"
echo "Results: ${RESULTS_DIR}/"
echo "=========================================="
