#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=true

BASE_RUN_ROOT="${RUN_ROOT:-/home/ubuntu/sn15_share_dir/Process_Q_Model_runs}"
AUTHOR_RUN_ROOT="${AUTHOR_RUN_ROOT:-${BASE_RUN_ROOT}/author_ckpt_eval}"
BACKBONE_PATH="${BACKBONE_PATH:-/home/ubuntu/shared-data/LLM/model/deepseek-math-7b-base}"
AUTHOR_CKPT="${AUTHOR_CKPT:-/home/ubuntu/sn15_share_dir/Process_Q_Model_runs/checkpoints/PQM/zeta-4/model.safetensors}"
ANALYZE="${ANALYZE:-1}"
SUMMARY="${SUMMARY:-1}"

echo "Evaluating official checkpoint: ${AUTHOR_CKPT}"
echo "Using backbone:                 ${BACKBONE_PATH}"
echo "Writing outputs to:             ${AUTHOR_RUN_ROOT}/outputs"

if [[ ! -f "${AUTHOR_CKPT}" ]]; then
  echo "AUTHOR_CKPT not found: ${AUTHOR_CKPT}" >&2
  exit 1
fi

if [[ ! -d "${BACKBONE_PATH}" ]]; then
  echo "BACKBONE_PATH not found: ${BACKBONE_PATH}" >&2
  exit 1
fi

CKPT_ROOT="${CKPT_ROOT:-unused}" \
MODEL_CKPT="${AUTHOR_CKPT}" \
BACKBONE_PATH="${BACKBONE_PATH}" \
SUMMARY_AFTER_EACH=1 \
RUN_ROOT="${AUTHOR_RUN_ROOT}" \
bash scripts/run_all_eval_2xa800.sh

if [[ "${ANALYZE}" == "1" ]]; then
  echo "Running reward diagnostics..."

  analyze_math() {
    local policy="$1"
    local scored_file="${AUTHOR_RUN_ROOT}/outputs/math-${policy}-scored.json"
    local report_file="${AUTHOR_RUN_ROOT}/outputs/math-${policy}-diagnostics.txt"

    if [[ -f "${scored_file}" ]]; then
      python scripts/analyze_scored_rewards.py \
        --scored-file "${scored_file}" \
        --dataset-name math \
        --original-file data/MATH-500/test.jsonl \
        | tee "${report_file}"
    fi
  }

  analyze_gsm() {
    local policy="$1"
    local scored_file="${AUTHOR_RUN_ROOT}/outputs/gsm-${policy}-scored.json"
    local report_file="${AUTHOR_RUN_ROOT}/outputs/gsm-${policy}-diagnostics.txt"

    if [[ -f "${scored_file}" ]]; then
      python scripts/analyze_scored_rewards.py \
        --scored-file "${scored_file}" \
        --dataset-name gsm8k \
        --original-file data/GSM-Plus/data/testmini-00000-of-00001.jsonl \
        | tee "${report_file}"
    fi
  }

  analyze_math "metamath-mistral"
  analyze_math "muggle"
  analyze_math "llama3-70b-inst"
  analyze_gsm "metamath-mistral"
  analyze_gsm "muggle"
  analyze_gsm "llama3-70b-inst"
fi

if [[ "${SUMMARY}" == "1" ]]; then
  echo "Writing compact summary table..."
  python scripts/summarize_eval_outputs.py \
    --outputs-dir "${AUTHOR_RUN_ROOT}/outputs" \
    --math-file data/MATH-500/test.jsonl \
    --gsm-file data/GSM-Plus/data/testmini-00000-of-00001.jsonl \
    --output-md "${AUTHOR_RUN_ROOT}/outputs/eval_summary.md" \
    --output-txt "${AUTHOR_RUN_ROOT}/outputs/eval_summary.txt"
fi

echo "Author checkpoint evaluation finished."
echo "Scored files, diagnostics, and summary tables are in: ${AUTHOR_RUN_ROOT}/outputs"
