#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${1:-/media/pc/D/datasets/vkitti2}
OUTPUT_DIR=${2:-outputs/vkitti_run}
CONFIG_PATH=${3:-configs/model_config.yaml}
PROMPT_CONFIG=${4:-configs/prompt_config.yaml}
CAMERA=${5:-Camera_0}

PROMPT_ARGS=()
if [[ -n "${PROMPT_CONFIG}" && -f "${PROMPT_CONFIG}" ]]; then
  PROMPT_ARGS+=(--prompt-config "${PROMPT_CONFIG}")
fi

python -m src.inference_pipeline vkitti \
  --input "${DATASET_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --config "${CONFIG_PATH}" \
  --vkitti-camera "${CAMERA}" \
  "${PROMPT_ARGS[@]}"
