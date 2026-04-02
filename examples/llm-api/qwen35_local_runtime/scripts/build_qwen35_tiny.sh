#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT}/../../.." && pwd)"
IMAGE="${TRTLLM_QWEN35_RUNTIME_IMAGE:-local/trtllm-qwen35-main-runtime:25.06}"
SRC_MODEL_DIR="${QWEN35_HOST_MODEL_DIR:-/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B}"
MIRROR_DIR="${ROOT}/artifacts/Qwen3.5-0.8B-text-mirror"
WORKSPACE_DIR="${ROOT}/artifacts/bench_workspace"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/build_qwen35_tiny.log"

mkdir -p "${LOG_DIR}"
QWEN35_VLM_SRC_DIR="${SRC_MODEL_DIR}" \
QWEN35_TEXT_MIRROR_DST_DIR="${MIRROR_DIR}" \
python3 "${ROOT}/scripts/normalize_qwen35_vlm_checkpoint.py" >/dev/null
rm -rf "${WORKSPACE_DIR}"

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_DIR}:/workspace/src" \
  -v "${ROOT}:/workspace/runtime" \
  -v "${SRC_MODEL_DIR}:/workspace/models/Qwen3.5-0.8B:ro" \
  "${IMAGE}" \
  bash -lc "\
    export PYTHONPATH=/workspace/src:\${PYTHONPATH}; \
    export LD_LIBRARY_PATH=/workspace/src/tensorrt_llm/libs:\${LD_LIBRARY_PATH}; \
    python -m tensorrt_llm.commands.bench \
      --model Qwen/Qwen3.5-0.8B \
      --model_path /workspace/runtime/artifacts/Qwen3.5-0.8B-text-mirror \
      --workspace /workspace/runtime/artifacts/bench_workspace \
      --log_level info \
      build \
      --max_seq_len 127 \
      --max_batch_size 1 \
      --max_num_tokens 127 \
      --trust_remote_code true" \
  2>&1 | tee "${LOG_FILE}"

echo "engine_dir=${WORKSPACE_DIR}/Qwen/Qwen3.5-0.8B/tp_1_pp_1"
echo "log_file=${LOG_FILE}"
