#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH=${1:?"Missing input image path"}
OUTPUT_DIR=${2:-outputs/image_run}
CONFIG_PATH=${3:-configs/model_config.yaml}
PROMPT_CONFIG=${4:-}

PROMPT_ARGS=()
if [[ -n "${PROMPT_CONFIG}" ]]; then
  PROMPT_ARGS+=(--prompt-config "${PROMPT_CONFIG}")
fi

python -m src.inference_pipeline image \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_DIR}" \
  --config "${CONFIG_PATH}" \
  "${PROMPT_ARGS[@]}"
