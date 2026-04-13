#!/bin/bash
#SBATCH --job-name=rhythmic_opt
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=72:00:00
#SBATCH --output=slurm_rhythmic_opt_%j.out
#SBATCH --error=slurm_rhythmic_opt_%j.err

# ============================================================
# OPTIMIZED Cross-Domain Experiments (Full: GG, RR, GR, RG)
# ============================================================
# Optimization: Train model ONCE per SNR level, then eval-only for others
#
# Without optimization: 4 types × 6 train_snr × 6 test_snr = 144 full trainings
# With optimization:    4 types × 6 train_snr = 24 trainings + 120 eval-only
#
# Speedup: ~6x faster (eval-only is ~instant vs 15 epochs training)
#
# Usage:
#   sbatch scripts/train_cross_domain_rhythmic_optimized.sh          # Uses run_02 by default
#   RUN_ID=run_03 sbatch scripts/train_cross_domain_rhythmic_optimized.sh
# ============================================================

set -e
cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# Run ID for organizing multiple experiment runs (default: run_02)
RUN_ID="${RUN_ID:-run_02}"

RHYTHMIC_DIR="autrainer-configurations/data/AudioSet-Balanced-Rhythmic"
RESULTS_DIR="results_rhythmic/${RUN_ID}"
MODELS_DIR="trained_models/${RUN_ID}"

SNR_LEVELS=("-5" "0" "10" "20" "30" "40")

mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "OPTIMIZED Cross-Domain Experiments"
echo "Run ID: $RUN_ID"
echo "Rhythmic Dataset: $RHYTHMIC_DIR"
echo "Results: $RESULTS_DIR"
echo "Models cache: $MODELS_DIR"
echo "=========================================="
nvidia-smi
echo "=========================================="

START_TIME=$(date +%s)
TRAIN_COUNT=0
EVAL_COUNT=0
SKIP_COUNT=0

log_train() {
    TRAIN_COUNT=$((TRAIN_COUNT + 1))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "[TRAIN $TRAIN_COUNT] $1"
    echo "Elapsed: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m"
    echo "=========================================="
}

log_eval() {
    EVAL_COUNT=$((EVAL_COUNT + 1))
    echo "[EVAL $EVAL_COUNT] $1"
}

log_skip() {
    SKIP_COUNT=$((SKIP_COUNT + 1))
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

# Save model symlink after training
save_model_link() {
    local exp_type=$1 train_snr=$2
    local run_dir="${RESULTS_DIR}/${exp_type}_train${train_snr}_test${train_snr}/training"
    local model_dir=$(ls -d "${run_dir}"/TIMIT* 2>/dev/null | head -1)
    if [ -n "$model_dir" ] && [ -f "${model_dir}/_best/model.pt" ]; then
        ln -sf "$(pwd)/${model_dir}/_best/model.pt" "${MODELS_DIR}/${exp_type}_${train_snr}dB.pt"
        echo "  Saved model: ${MODELS_DIR}/${exp_type}_${train_snr}dB.pt"
    fi
}

get_model_path() {
    local exp_type=$1 train_snr=$2
    local link="${MODELS_DIR}/${exp_type}_${train_snr}dB.pt"
    if [ -L "$link" ] && [ -e "$link" ]; then
        echo "$link"
    fi
}

# ============================================================
# CONFIG GENERATORS
# ============================================================

# Generate TRAINING config (iterations=15)
gen_train_config() {
    local cfg=$1 exp=$2 train_type=$3 train_snr=$4 test_type=$5 test_snr=$6
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""

    # Train augmentation
    if [ "$train_type" = "G" ]; then
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

    # Test augmentation
    if [ "$test_type" = "G" ]; then
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
    echo "  Generated TRAIN config: $cfg"
}

# Generate EVAL-ONLY config (iterations=0, load checkpoint)
gen_eval_config() {
    local cfg=$1 exp=$2 train_type=$3 train_snr=$4 test_type=$5 test_snr=$6 model_path=$7
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""

    # Train augmentation (for naming only, not used)
    if [ "$train_type" = "G" ]; then
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

    # Test augmentation
    if [ "$test_type" = "G" ]; then
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
iterations: 0
hydra:
  sweeper:
    params:
      +seed: 0
      +batch_size: 64
      +learning_rate: 0.001
      dataset: TIMIT-sentencetype-16k
      model: Cnn10-32k-T
      +model.model_checkpoint: ${model_path}
      +model.skip_last_layer: false
      optimizer: Adam
${train_aug}
${test_aug}
EOF
    echo "  Generated EVAL config: $cfg"
}

run_train() {
    local cfg=$1 exp=$2 desc=$3
    check_exists "$exp" && { log_skip "$desc (exists)"; return 0; }
    log_train "$desc"
    autrainer train -cn "${cfg}.yaml"
}

run_eval() {
    local cfg=$1 exp=$2 desc=$3
    check_exists "$exp" && { log_skip "$desc (exists)"; return 0; }
    log_eval "$desc"
    autrainer train -cn "${cfg}.yaml"
}

# ============================================================
echo ""
echo "=========================================="
echo "PHASE 1: Training base models (diagonal)"
echo "One training per SNR level per noise type"
echo "=========================================="
# ============================================================

# GG: Gaussian train -> Gaussian test (diagonal)
echo ""
echo "--- GG: Training diagonal ---"
for snr in "${SNR_LEVELS[@]}"; do
    cfg="R_GG_train${snr}dB_test${snr}dB"
    exp="GG_train${snr}_test${snr}"
    gen_train_config "$cfg" "$exp" "G" "$snr" "G" "$snr"
    run_train "$cfg" "$exp" "GG train@${snr}dB -> test@${snr}dB"
    save_model_link "GG" "$snr"
done

# RR: Rhythmic train -> Rhythmic test (diagonal)
echo ""
echo "--- RR: Training diagonal ---"
for snr in "${SNR_LEVELS[@]}"; do
    cfg="R_RR_train${snr}dB_test${snr}dB"
    exp="RR_train${snr}_test${snr}"
    gen_train_config "$cfg" "$exp" "R" "$snr" "R" "$snr"
    run_train "$cfg" "$exp" "RR train@${snr}dB -> test@${snr}dB"
    save_model_link "RR" "$snr"
done

# ============================================================
echo ""
echo "=========================================="
echo "PHASE 2: Evaluation-only (off-diagonal)"
echo "Reusing trained models with iterations=0"
echo "=========================================="
# ============================================================

# GG off-diagonal
echo ""
echo "--- GG: Evaluating off-diagonal ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "GG" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No GG model for ${train_snr}dB"; continue; }
    
    for test_snr in "${SNR_LEVELS[@]}"; do
        [ "$train_snr" = "$test_snr" ] && continue
        cfg="R_GG_train${train_snr}dB_test${test_snr}dB"
        exp="GG_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "G" "$train_snr" "G" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "GG ${train_snr}->${test_snr}dB"
    done
done

# RR off-diagonal
echo ""
echo "--- RR: Evaluating off-diagonal ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "RR" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No RR model for ${train_snr}dB"; continue; }
    
    for test_snr in "${SNR_LEVELS[@]}"; do
        [ "$train_snr" = "$test_snr" ] && continue
        cfg="R_RR_train${train_snr}dB_test${test_snr}dB"
        exp="RR_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "R" "$train_snr" "R" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "RR ${train_snr}->${test_snr}dB"
    done
done

# GR: Gaussian train -> Rhythmic test (all combinations, eval-only using GG models)
echo ""
echo "--- GR: Gaussian train -> Rhythmic test (eval-only) ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "GG" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No GG model for ${train_snr}dB"; continue; }
    
    for test_snr in "${SNR_LEVELS[@]}"; do
        cfg="R_GR_train${train_snr}dB_test${test_snr}dB"
        exp="GR_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "G" "$train_snr" "R" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "GR G${train_snr}->R${test_snr}dB"
    done
done

# RG: Rhythmic train -> Gaussian test (all combinations, eval-only using RR models)
echo ""
echo "--- RG: Rhythmic train -> Gaussian test (eval-only) ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "RR" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No RR model for ${train_snr}dB"; continue; }
    
    for test_snr in "${SNR_LEVELS[@]}"; do
        cfg="R_RG_train${train_snr}dB_test${test_snr}dB"
        exp="RG_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "R" "$train_snr" "G" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "RG R${train_snr}->G${test_snr}dB"
    done
done

# ============================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "COMPLETED!"
echo "=========================================="
echo "Trained: $TRAIN_COUNT models (15 epochs each)"
echo "Evaluated: $EVAL_COUNT experiments (iterations=0, instant)"
echo "Skipped: $SKIP_COUNT experiments (already exist)"
echo "Total time: $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m"
echo ""
echo "Optimization stats:"
echo "  Without optimization: 144 full trainings"
echo "  With optimization: 12 trainings + 132 eval-only"
echo "  Estimated speedup: ~6x"
echo "=========================================="
