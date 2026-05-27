#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=true

RUN_ROOT="${RUN_ROOT:-/home/ubuntu/sn15_share_dir/Process_Q_Model_runs}"
mkdir -p "${RUN_ROOT}"/{checkpoints,tmp,cache,triton,torch_extensions}
export HF_HOME="${HF_HOME:-${RUN_ROOT}/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${RUN_ROOT}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUN_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUN_ROOT}/torch_extensions}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/shared-data/LLM/model/deepseek-math-7b-base}"
DATASET_PATH="${DATASET_PATH:-data/Math-Shepherd}"
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints/pqm-zeta4-2xa800}"
MASTER_PORT="${MASTER_PORT:-29501}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
# Empirically, GA4 matches the released PQM checkpoint behavior much better on 2xA800.
# GA16 only matches the 8-GPU optimizer-step count, but produced a broken reward ranking.
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RESUME_ARG=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARG=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPU_LIST[@]}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${#VISIBLE_GPU_LIST[@]} GPU(s), but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  echo "Use CUDA_VISIBLE_DEVICES=0,1 for 2-GPU training, or set NPROC_PER_NODE=1 for a single-GPU run." >&2
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_main.py \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --save-path "${SAVE_PATH}" \
  --deepspeed-config accelerate_configs/deepspeed_3_no_offload.json \
  --save-strategy epoch \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --loss-type rank \
  --zeta 4 \
  "${RESUME_ARG[@]}"
