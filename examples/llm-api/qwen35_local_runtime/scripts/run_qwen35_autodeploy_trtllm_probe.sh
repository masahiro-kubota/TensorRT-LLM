#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/../../.." && pwd)"
IMAGE="${TRTLLM_QWEN35_RUNTIME_IMAGE:-local/trtllm-qwen35-main-runtime:25.06}"
SRC_MODEL_DIR="${QWEN35_HOST_MODEL_DIR:-/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B}"

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e TLLM_DISABLE_MPI=1 \
  -e PYTHONPATH="/workspace/src:${PYTHONPATH:-}" \
  -e LD_LIBRARY_PATH="/workspace/src/tensorrt_llm/libs:${LD_LIBRARY_PATH:-}" \
  -e QWEN35_VLM_SRC_DIR="/workspace/models/Qwen3.5-0.8B" \
  -e QWEN35_TEXT_CLEAN_DST_DIR="/workspace/runtime/artifacts/Qwen3.5-0.8B-text-clean" \
  -v "${REPO_DIR}:/workspace/src" \
  -v "${ROOT_DIR}:/workspace/runtime" \
  -v "${SRC_MODEL_DIR}:/workspace/models/Qwen3.5-0.8B:ro" \
  "${IMAGE}" \
  bash -lc "python -c \"import importlib.util, subprocess, sys; importlib.util.find_spec('graphviz') or subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'graphviz'])\" && cd /workspace/src && python /workspace/runtime/scripts/build_qwen35_text_clean_mirror.py && python /workspace/runtime/scripts/qwen35_autodeploy_trtllm_probe.py"
