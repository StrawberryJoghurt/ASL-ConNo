#!/bin/bash
#SBATCH --job-name=cd_model_opt
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=72:00:00
#SBATCH --output=slurm_cd_model_opt_%j.out
#SBATCH --error=slurm_cd_model_opt_%j.err

set -euo pipefail

cd /data/chi-gpu1/go29hoq/tum_practical_project/ASL-ConNo
source .venv/bin/activate

# ============================================================
# Parameterized optimized cross-domain sweep (Gaussian + AudioSet)
# ============================================================
# Default target:
#   MODEL_ID=ConvNeXt-Tiny-T
#   OPTIMIZER_ID=AdamW
#   BATCH_SIZE=16
#   LEARNING_RATE=0.0001
#
# Smoke test:
#   sbatch --export=ALL,MODEL_ID=ConvNeXt-Tiny-T,OPTIMIZER_ID=AdamW,BATCH_SIZE=16,LEARNING_RATE=0.0001,ITERATIONS=1,SNR_LEVELS_STR=\"40\" scripts/train_cross_domain_model_optimized.sh
#
# Full sweep:
#   sbatch --export=ALL,MODEL_ID=ConvNeXt-Tiny-T,OPTIMIZER_ID=AdamW,BATCH_SIZE=16,LEARNING_RATE=0.0001 scripts/train_cross_domain_model_optimized.sh
# ============================================================

MODEL_ID="${MODEL_ID:-ConvNeXt-Tiny-T}"
OPTIMIZER_ID="${OPTIMIZER_ID:-AdamW}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
ITERATIONS="${ITERATIONS:-15}"
SNR_LEVELS_STR="${SNR_LEVELS_STR:--5 0 10 20 30 40}"
AUDIOSET_DIR="${AUDIOSET_DIR:-autrainer-configurations/data/AudioSet-Balanced-Noise}"

MODEL_SLUG="${MODEL_SLUG:-$(printf '%s' "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//')}"
RUN_ID="${RUN_ID:-${MODEL_SLUG}_run_01}"
RESULTS_DIR="results/${RUN_ID}"
MODELS_DIR="trained_models/${MODEL_SLUG}_${RUN_ID}"
CONFIG_PREFIX="${CONFIG_PREFIX:-M_${MODEL_SLUG}}"

read -r -a SNR_LEVELS <<< "$SNR_LEVELS_STR"

mkdir -p "$MODELS_DIR" "$RESULTS_DIR"

echo "=========================================="
echo "Parameterized Cross-Domain Experiments"
echo "Model: $MODEL_ID"
echo "Model slug: $MODEL_SLUG"
echo "Optimizer: $OPTIMIZER_ID"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Iterations: $ITERATIONS"
echo "Run ID: $RUN_ID"
echo "AudioSet Dataset: $AUDIOSET_DIR"
echo "Results: $RESULTS_DIR"
echo "Models cache: $MODELS_DIR"
echo "SNR levels: ${SNR_LEVELS[*]}"
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

save_model_link() {
    local exp_type=$1 train_snr=$2
    local run_dir="${RESULTS_DIR}/${exp_type}_train${train_snr}_test${train_snr}/training"
    local model_dir
    model_dir=$(ls -d "${run_dir}"/TIMIT* 2>/dev/null | head -1)
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

gen_train_config() {
    local cfg=$1 exp=$2 train_type=$3 train_snr=$4 test_type=$5 test_snr=$6
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""

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
  id: AudioSetNoise(${train_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${AUDIOSET_DIR}
      noise_csv: ${AUDIOSET_DIR}/train.csv
      snr_db: ${train_snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null
      p: 1.0
      generator_seed: 0"
    fi

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
  id: AudioSetNoise(${test_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${AUDIOSET_DIR}
      noise_csv: ${AUDIOSET_DIR}/test.csv
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
iterations: ${ITERATIONS}
hydra:
  sweeper:
    params:
      +seed: 0
      +batch_size: ${BATCH_SIZE}
      +learning_rate: ${LEARNING_RATE}
      dataset: TIMIT-sentencetype-16k
      model: ${MODEL_ID}
      optimizer: ${OPTIMIZER_ID}
${train_aug}
${test_aug}
EOF
    echo "  Generated TRAIN config: $cfg"
}

gen_eval_config() {
    local cfg=$1 exp=$2 train_type=$3 train_snr=$4 test_type=$5 test_snr=$6 model_path=$7
    local f="conf/${cfg}.yaml"
    [ -f "$f" ] && return

    local train_aug="" test_aug=""

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
  id: AudioSetNoise(${train_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${AUDIOSET_DIR}
      noise_csv: ${AUDIOSET_DIR}/train.csv
      snr_db: ${train_snr}.0
      sample_rate: 16000
      n_fft: 512
      hop_length: 160
      n_mels: 64
      noise_type: null
      p: 1.0
      generator_seed: 0"
    fi

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
  id: AudioSetNoise(${test_snr}dB)
  pipeline:
  - CrossDomainNoise:
      _target_: autrainer.augmentations.spectrogram_augmentations.CrossDomainNoise
      noise_dir: ${AUDIOSET_DIR}
      noise_csv: ${AUDIOSET_DIR}/test.csv
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
      +batch_size: ${BATCH_SIZE}
      +learning_rate: ${LEARNING_RATE}
      dataset: TIMIT-sentencetype-16k
      model: ${MODEL_ID}
      +model.model_checkpoint: ${model_path}
      +model.skip_last_layer: false
      optimizer: ${OPTIMIZER_ID}
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

echo ""
echo "=========================================="
echo "PHASE 1: Training base models (diagonal)"
echo "=========================================="

echo ""
echo "--- GG: Training diagonal ---"
for snr in "${SNR_LEVELS[@]}"; do
    cfg="${CONFIG_PREFIX}_GG_train${snr}dB_test${snr}dB"
    exp="GG_train${snr}_test${snr}"
    gen_train_config "$cfg" "$exp" "G" "$snr" "G" "$snr"
    run_train "$cfg" "$exp" "GG train@${snr}dB -> test@${snr}dB"
    save_model_link "GG" "$snr"
done

echo ""
echo "--- AA: Training diagonal ---"
for snr in "${SNR_LEVELS[@]}"; do
    cfg="${CONFIG_PREFIX}_AA_train${snr}dB_test${snr}dB"
    exp="AA_train${snr}_test${snr}"
    gen_train_config "$cfg" "$exp" "A" "$snr" "A" "$snr"
    run_train "$cfg" "$exp" "AA train@${snr}dB -> test@${snr}dB"
    save_model_link "AA" "$snr"
done

echo ""
echo "=========================================="
echo "PHASE 2: Evaluation-only (off-diagonal)"
echo "=========================================="

echo ""
echo "--- GG: Evaluating off-diagonal ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "GG" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No GG model for ${train_snr}dB"; continue; }

    for test_snr in "${SNR_LEVELS[@]}"; do
        [ "$train_snr" = "$test_snr" ] && continue
        cfg="${CONFIG_PREFIX}_GG_train${train_snr}dB_test${test_snr}dB"
        exp="GG_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "G" "$train_snr" "G" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "GG ${train_snr}->${test_snr}dB"
    done
done

echo ""
echo "--- AA: Evaluating off-diagonal ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "AA" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No AA model for ${train_snr}dB"; continue; }

    for test_snr in "${SNR_LEVELS[@]}"; do
        [ "$train_snr" = "$test_snr" ] && continue
        cfg="${CONFIG_PREFIX}_AA_train${train_snr}dB_test${test_snr}dB"
        exp="AA_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "A" "$train_snr" "A" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "AA ${train_snr}->${test_snr}dB"
    done
done

echo ""
echo "--- GA: Gaussian train -> AudioSet test (eval-only) ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "GG" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No GG model for ${train_snr}dB"; continue; }

    for test_snr in "${SNR_LEVELS[@]}"; do
        cfg="${CONFIG_PREFIX}_GA_train${train_snr}dB_test${test_snr}dB"
        exp="GA_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "G" "$train_snr" "A" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "GA G${train_snr}->A${test_snr}dB"
    done
done

echo ""
echo "--- AG: AudioSet train -> Gaussian test (eval-only) ---"
for train_snr in "${SNR_LEVELS[@]}"; do
    model_path=$(get_model_path "AA" "$train_snr")
    [ -z "$model_path" ] && { echo "WARNING: No AA model for ${train_snr}dB"; continue; }

    for test_snr in "${SNR_LEVELS[@]}"; do
        cfg="${CONFIG_PREFIX}_AG_train${train_snr}dB_test${test_snr}dB"
        exp="AG_train${train_snr}_test${test_snr}"
        gen_eval_config "$cfg" "$exp" "A" "$train_snr" "G" "$test_snr" "$model_path"
        run_eval "$cfg" "$exp" "AG A${train_snr}->G${test_snr}dB"
    done
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "COMPLETED!"
echo "=========================================="
echo "Model: $MODEL_ID"
echo "Run ID: $RUN_ID"
echo "Trained: $TRAIN_COUNT models"
echo "Evaluated: $EVAL_COUNT experiments"
echo "Skipped: $SKIP_COUNT experiments"
echo "Total time: $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m"
echo "=========================================="
