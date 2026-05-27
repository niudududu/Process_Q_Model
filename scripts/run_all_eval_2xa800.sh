#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=true

RUN_ROOT="${RUN_ROOT:-/home/ubuntu/sn15_share_dir/Process_Q_Model_runs}"
BACKBONE_PATH="${BACKBONE_PATH:-/home/ubuntu/shared-data/LLM/model/deepseek-math-7b-base}"
CKPT_ROOT="${CKPT_ROOT:-${RUN_ROOT}/checkpoints/pqm-zeta4-2xa800}"
MODEL_CKPT="${MODEL_CKPT:-}"
SUMMARY_AFTER_EACH="${SUMMARY_AFTER_EACH:-0}"

mkdir -p "${RUN_ROOT}"/{outputs,tmp,cache,triton,torch_extensions}
export HF_HOME="${HF_HOME:-${RUN_ROOT}/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${RUN_ROOT}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUN_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUN_ROOT}/torch_extensions}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp}"

if [[ -z "${MODEL_CKPT}" ]]; then
  LATEST_CKPT="$(ls -d "${CKPT_ROOT}"/checkpoint-* | sort -V | tail -1)"
  MODEL_CKPT="${LATEST_CKPT}/pytorch_model.bin"
fi

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "MODEL_CKPT not found: ${MODEL_CKPT}" >&2
  exit 1
fi

echo "Using checkpoint: ${MODEL_CKPT}"
echo "Using backbone:   ${BACKBONE_PATH}"
echo "Outputs:          ${RUN_ROOT}/outputs"

write_summary() {
  if [[ "${SUMMARY_AFTER_EACH}" != "1" ]]; then
    return 0
  fi

  python scripts/summarize_eval_outputs.py \
    --outputs-dir "${RUN_ROOT}/outputs" \
    --math-file data/MATH-500/test.jsonl \
    --gsm-file data/GSM-Plus/data/testmini-00000-of-00001.jsonl \
    --output-md "${RUN_ROOT}/outputs/eval_summary.md" \
    --output-txt "${RUN_ROOT}/outputs/eval_summary.txt"
}

run_math() {
  local policy="$1"
  local data_file="$2"
  local save_file="${RUN_ROOT}/outputs/math-${policy}-scored.json"

  echo "==== MATH / ${policy} ===="
  BACKBONE_PATH="${BACKBONE_PATH}" \
  MODEL_CKPT="${MODEL_CKPT}" \
  DATA_FILE="${data_file}" \
  MATH_FILE="data/MATH-500/test.jsonl" \
  SAVE_FILE="${save_file}" \
  bash scripts/run_eval_math_2xa800.sh
  write_summary
}

run_gsm() {
  local policy="$1"
  local data_file="$2"
  local save_file="${RUN_ROOT}/outputs/gsm-${policy}-scored.json"

  echo "==== GSM-Plus / ${policy} ===="
  BACKBONE_PATH="${BACKBONE_PATH}" \
  MODEL_CKPT="${MODEL_CKPT}" \
  DATA_FILE="${data_file}" \
  GSM_PLUS_FILE="data/GSM-Plus/data/testmini-00000-of-00001.jsonl" \
  SAVE_FILE="${save_file}" \
  bash scripts/run_eval_gsm_2xa800.sh
  write_summary
}

run_math "metamath-mistral" "data/PQM/eval_data/math-metamath-mistral-128.json"
run_math "muggle" "data/PQM/eval_data/math-muggle-128.json"
run_math "llama3-70b-inst" "data/PQM/eval_data/math-llama3-70b-inst-128.json"

run_gsm "metamath-mistral" "data/PQM/eval_data/gsm8k-plus-metamath-mistral-128.json"
run_gsm "muggle" "data/PQM/eval_data/gsm8k-plus-muggle-128.json"
run_gsm "llama3-70b-inst" "data/PQM/eval_data/gsm8k-plus-llama3-70b-inst-128.json"

echo "All evaluations finished."
