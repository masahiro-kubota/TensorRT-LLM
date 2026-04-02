#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/../../.." && pwd)"
IMAGE="${TRTLLM_QWEN35_RUNTIME_IMAGE:-local/trtllm-qwen35-main-runtime:25.06}"
SRC_MODEL_DIR="${QWEN35_HOST_MODEL_DIR:-/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B}"
SCRIPT_PATH="/workspace/runtime/scripts/qwen35_pytorch_backend_probe.py"

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e TLLM_DISABLE_MPI=1 \
  -e PYTHONPATH="/workspace/src:${PYTHONPATH:-}" \
  -e LD_LIBRARY_PATH="/workspace/src/tensorrt_llm/libs:${LD_LIBRARY_PATH:-}" \
  -e QWEN35_VLM_SRC_DIR="/workspace/models/Qwen3.5-0.8B" \
  -e QWEN35_TEXT_MIRROR_DST_DIR="/workspace/runtime/artifacts/Qwen3.5-0.8B-text-mirror" \
  -v "${REPO_DIR}:/workspace/src" \
  -v "${ROOT_DIR}:/workspace/runtime" \
  -v "${SRC_MODEL_DIR}:/workspace/models/Qwen3.5-0.8B:ro" \
  "${IMAGE}" \
  bash -lc "cd /workspace/src && python /workspace/runtime/scripts/normalize_qwen35_vlm_checkpoint.py && python ${SCRIPT_PATH}"
