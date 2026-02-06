#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train_ablation.py"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

GPUS_CSV="${GPUS:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if (( ${#GPUS[@]} < 2 )); then
  echo "GPUS must contain at least 2 GPU ids (example: GPUS=0,1)" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_PREFIX="${RUN_PREFIX:-contaccum_ablation}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/experiments/contaccum_ablation/logs/${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/contaccum_ablation/output/models}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

MODEL_NAME="${MODEL_NAME:-answerdotai/ModernBERT-base}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:--1}"
SEED="${SEED:-12}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
OPTIM="${OPTIM:-adamw_torch}"
LOSS_MINI_BATCH_SIZE="${LOSS_MINI_BATCH_SIZE:-128}"
BATCH_SAMPLER="${BATCH_SAMPLER:-no_duplicates}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-512}"

TARGET_BSZ="${TARGET_BSZ:-8192}"
GRADACCUM_MICRO_BSZ="${GRADACCUM_MICRO_BSZ:-1024}"
CONT_MICRO_BSZ="${CONT_MICRO_BSZ:-1024}"
CACHED_BSZ="${CACHED_BSZ:-8192}"

if (( TARGET_BSZ % GRADACCUM_MICRO_BSZ != 0 )); then
  echo "TARGET_BSZ must be divisible by GRADACCUM_MICRO_BSZ" >&2
  exit 1
fi
if (( TARGET_BSZ % CONT_MICRO_BSZ != 0 )); then
  echo "TARGET_BSZ must be divisible by CONT_MICRO_BSZ" >&2
  exit 1
fi

GRADACCUM_STEPS="${GRADACCUM_STEPS:-$((TARGET_BSZ / GRADACCUM_MICRO_BSZ))}"
CONT_GRAD_ACCUM_STEPS="${CONT_GRAD_ACCUM_STEPS:-1}"
CONT_BANK_SIZE="${CONT_BANK_SIZE:-$((TARGET_BSZ / CONT_MICRO_BSZ - 1))}"
CACHED_GRAD_ACCUM_STEPS="${CACHED_GRAD_ACCUM_STEPS:-1}"
CACHED_BANK_SIZE="${CACHED_BANK_SIZE:-${CONT_BANK_SIZE}}"

WANDB_PROJECT="${WANDB_PROJECT:-st-contaccum-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}" # online | offline | disabled
WANDB_GROUP="${WANDB_GROUP:-${RUN_PREFIX}_${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"

# Optional: set GATHER_ACROSS_DEVICES=1 if each run is multi-GPU/DDP.
GATHER_ACROSS_DEVICES="${GATHER_ACROSS_DEVICES:-0}"

run_one() {
  local gpu="$1"
  local direction="$2"
  local strategy="$3"
  local train_bsz="$4"
  local grad_accum="$5"
  local bank_size="$6"

  local run_name="${RUN_PREFIX}_${direction}_${strategy}_bs${train_bsz}_ga${grad_accum}_bank${bank_size}_${RUN_TAG}"
  local log_file="${LOG_ROOT}/${run_name}.log"

  echo "[launch] gpu=${gpu} run=${run_name}"

  local -a cmd=(
    "${PYTHON_BIN}" "${TRAIN_PY}"
    --model_name "${MODEL_NAME}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --max_train_examples "${MAX_TRAIN_EXAMPLES}"
    --seed "${SEED}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${train_bsz}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --weight_decay "${WEIGHT_DECAY}"
    --gradient_accumulation_steps "${grad_accum}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --optim "${OPTIM}"
    --loss_mini_batch_size "${LOSS_MINI_BATCH_SIZE}"
    --batch_sampler "${BATCH_SAMPLER}"
    --direction "${direction}"
    --strategy "${strategy}"
    --bank_size "${bank_size}"
    --output_root "${OUTPUT_ROOT}"
    --run_name "${run_name}"
    --max_steps "${MAX_STEPS}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
    --wandb_group "${WANDB_GROUP}"
  )

  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ "${GATHER_ACROSS_DEVICES}" == "1" ]]; then
    cmd+=(--gather_across_devices)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf "[dry-run] CUDA_VISIBLE_DEVICES=%s " "${gpu}"
      printf "%q " "${cmd[@]}"
      printf "\n"
    } | tee "${log_file}"
    return 0
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export WANDB_PROJECT="${WANDB_PROJECT}"
    export WANDB_MODE="${WANDB_MODE}"
    if [[ -n "${WANDB_ENTITY}" ]]; then
      export WANDB_ENTITY="${WANDB_ENTITY}"
    fi
    "${cmd[@]}"
  ) 2>&1 | tee "${log_file}"
}

declare -a EXPERIMENTS=(
  "uni grad_accum ${GRADACCUM_MICRO_BSZ} ${GRADACCUM_STEPS} 0"
  "bi grad_accum ${GRADACCUM_MICRO_BSZ} ${GRADACCUM_STEPS} 0"
  "uni cont_accum ${CONT_MICRO_BSZ} ${CONT_GRAD_ACCUM_STEPS} ${CONT_BANK_SIZE}"
  "bi cont_accum ${CONT_MICRO_BSZ} ${CONT_GRAD_ACCUM_STEPS} ${CONT_BANK_SIZE}"
  "uni cached_cont_accum ${CACHED_BSZ} ${CACHED_GRAD_ACCUM_STEPS} ${CACHED_BANK_SIZE}"
  "bi cached_cont_accum ${CACHED_BSZ} ${CACHED_GRAD_ACCUM_STEPS} ${CACHED_BANK_SIZE}"
)

echo "[info] logs: ${LOG_ROOT}"
echo "[info] output_root: ${OUTPUT_ROOT}"
echo "[info] wandb_project: ${WANDB_PROJECT}, wandb_mode: ${WANDB_MODE}"
echo "[info] using gpus: ${GPUS[*]}"
echo "[info] dry_run: ${DRY_RUN}"

idx=0
total="${#EXPERIMENTS[@]}"
while (( idx < total )); do
  pids=()
  for slot in 0 1; do
    if (( idx >= total )); then
      break
    fi
    read -r direction strategy train_bsz grad_accum bank_size <<< "${EXPERIMENTS[$idx]}"
    run_one "${GPUS[$slot]}" "${direction}" "${strategy}" "${train_bsz}" "${grad_accum}" "${bank_size}" &
    pids+=("$!")
    idx=$((idx + 1))
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

echo "[done] all experiments completed"
