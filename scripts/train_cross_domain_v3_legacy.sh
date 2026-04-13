#!/bin/bash
#SBATCH --job-name=cross_domain_v3
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=96:00:00
#SBATCH --output=slurm_cross_domain_v3_%j.out
#SBATCH --error=slurm_cross_domain_v3_%j.err

# ============================================================
# Cross-Domain Noise Experiments - Version 3
# ============================================================
# FIXED: AudioSet uses CrossDomainNoise (not SNR_noise)
# NEW: Added 30dB and 40dB levels
# SNR levels: -5dB, 0dB, 10dB, 20dB, 30dB, 40dB
# 
# Usage:
#   sbatch scripts/train_cross_domain_v3.sh          # Uses run_02 by default
#   RUN_ID=run_03 sbatch scripts/train_cross_domain_v3.sh
# ============================================================

cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Run ID for organizing multiple experiment runs (default: run_02)
RUN_ID="${RUN_ID:-run_02}"
RESULTS_DIR="results/${RUN_ID}"

mkdir -p "$RESULTS_DIR"

SNR_LEVELS=("-5" "0" "10" "20" "30" "40")

echo "=========================================="
echo "Cross-Domain Noise Experiments v3"
echo "Run ID: ${RUN_ID}"
echo "Results Dir: ${RESULTS_DIR}"
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

# Generate AudioSet config (CrossDomainNoise)
gen_audioset() {
    local cfg=$1 exp=$2 train_snr=$3 test_snr=$4
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""
    if [ -n "$train_snr" ]; then
        train_aug="train_augmentation:
  _target_: autrainer.augmentations.augmentation_pipeline.AugmentationPipeline
  id: AudioSetNoise(${train_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: autrainer-configurations/data/AudioSet-Balanced-Noise
      noise_csv: autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv
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
  id: AudioSetNoise(${test_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: autrainer-configurations/data/AudioSet-Balanced-Noise
      noise_csv: autrainer-configurations/data/AudioSet-Balanced-Noise/test.csv
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

# Generate cross-domain config
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
  id: AudioSetNoise(${train_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: autrainer-configurations/data/AudioSet-Balanced-Noise
      noise_csv: autrainer-configurations/data/AudioSet-Balanced-Noise/train.csv
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
  id: AudioSetNoise(${test_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: autrainer-configurations/data/AudioSet-Balanced-Noise
      noise_csv: autrainer-configurations/data/AudioSet-Balanced-Noise/test.csv
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

# Baseline degradation
for s in "${SNR_LEVELS[@]}"; do
    gen_gaussian "CleanTrain_GaussianTest_${s}dB" "CleanTrain_Gaussian${s}" "" "$s"
    gen_audioset "CleanTrain_AudioSetTest_${s}dB" "CleanTrain_AudioSet${s}" "" "$s"
done

# GG
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_gaussian "GG_train${t}dB_test${e}dB" "GG_train${t}_test${e}" "$t" "$e"
    done
done

# AA
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_audioset "AA_train${t}dB_test${e}dB" "AA_train${t}_test${e}" "$t" "$e"
    done
done

# GA (Gaussian train -> AudioSet test)
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_cross "GA_train${t}dB_test${e}dB" "GA_train${t}_test${e}" "G" "$t" "A" "$e"
    done
done

# AG (AudioSet train -> Gaussian test)
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        gen_cross "AG_train${t}dB_test${e}dB" "AG_train${t}_test${e}" "A" "$t" "G" "$e"
    done
done

# Robustness
for s in "${SNR_LEVELS[@]}"; do
    gen_gaussian "Robust_Gaussian${s}dB_Clean" "Robust_G${s}_Clean" "$s" ""
    gen_audioset "Robust_AudioSet${s}dB_Clean" "Robust_A${s}_Clean" "$s" ""
done

echo "Config generation complete!"

# ============================================================
echo "Running experiments..."
# ============================================================

# Baseline
run_exp "Baseline_Clean" "baseline_clean" "Baseline Clean"

# Baseline degradation
for s in "${SNR_LEVELS[@]}"; do
    run_exp "CleanTrain_GaussianTest_${s}dB" "CleanTrain_Gaussian${s}" "Clean→Gaussian ${s}dB"
done
for s in "${SNR_LEVELS[@]}"; do
    run_exp "CleanTrain_AudioSetTest_${s}dB" "CleanTrain_AudioSet${s}" "Clean→AudioSet ${s}dB"
done

# GG
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "GG_train${t}dB_test${e}dB" "GG_train${t}_test${e}" "GG ${t}→${e}dB"
    done
done

# AA
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "AA_train${t}dB_test${e}dB" "AA_train${t}_test${e}" "AA ${t}→${e}dB"
    done
done

# GA
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "GA_train${t}dB_test${e}dB" "GA_train${t}_test${e}" "GA G${t}→A${e}dB"
    done
done

# AG
for t in "${SNR_LEVELS[@]}"; do
    for e in "${SNR_LEVELS[@]}"; do
        run_exp "AG_train${t}dB_test${e}dB" "AG_train${t}_test${e}" "AG A${t}→G${e}dB"
    done
done

# Robustness
for s in "${SNR_LEVELS[@]}"; do
    run_exp "Robust_Gaussian${s}dB_Clean" "Robust_G${s}_Clean" "Robust G${s}→Clean"
done
for s in "${SNR_LEVELS[@]}"; do
    run_exp "Robust_AudioSet${s}dB_Clean" "Robust_A${s}_Clean" "Robust A${s}→Clean"
done

# ============================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "COMPLETED!"
echo "Run: $EXPERIMENT_COUNT | Skipped: $SKIPPED_COUNT"
echo "Time: $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m"
echo "=========================================="
