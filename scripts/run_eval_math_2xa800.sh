#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=true

RUN_ROOT="${RUN_ROOT:-/home/ubuntu/sn15_share_dir/Process_Q_Model_runs}"
mkdir -p "${RUN_ROOT}"/{outputs,tmp,cache,triton,torch_extensions}
export HF_HOME="${HF_HOME:-${RUN_ROOT}/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${RUN_ROOT}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUN_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUN_ROOT}/torch_extensions}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp}"

BACKBONE_PATH="${BACKBONE_PATH:-/home/ubuntu/shared-data/LLM/model/deepseek-math-7b-base}"
MODEL_CKPT="${MODEL_CKPT:?Set MODEL_CKPT to a trained checkpoint file, for example checkpoints/pqm-zeta4-2xa800/checkpoint-*/pytorch_model.bin}"
DATA_FILE="${DATA_FILE:-data/PQM/eval_data/math-metamath-mistral-128.json}"
MATH_FILE="${MATH_FILE:-data/MATH-500/test.jsonl}"
SAVE_FILE="${SAVE_FILE:-${RUN_ROOT}/outputs/math-prm-scored.json}"

mkdir -p "$(dirname "${SAVE_FILE}")"

deepspeed --num_gpus=2 bon_eval_hf.py \
  --backbone-path "${BACKBONE_PATH}" \
  --model-path "${MODEL_CKPT}" \
  --data-name math \
  --data-file "${DATA_FILE}" \
  --math-file "${MATH_FILE}" \
  --save-file "${SAVE_FILE}"
