# Qwen3.5 on RTX 6000

This note explains how to reproduce the current local `Qwen3.5-0.8B` text-only
runtime path on an RTX 6000 machine from the remote branch.

## Short Answer

Cloning the branch is necessary, but not sufficient.

You also need:

- a TRT-LLM development container image
- a source build on top of this branch
- a local `Qwen3.5-0.8B` Hugging Face checkpoint

The current branch is:

- repo: `git@github.com:masahiro-kubota/TensorRT-LLM.git`
- branch: `codex/qwen35-wheel-build-fixes`

## What This Branch Can Do

After setup, this branch can reproduce:

- `backend=pytorch` on a normalized text-only mirror of `Qwen3.5-0.8B`
- AutoDeploy text-only probes on the normalized text-only mirror

This branch does not yet make the classic TensorRT engine build succeed for
dense Qwen3.5. `scripts/build_qwen35_tiny.sh` is still expected to fail on the
classic `tensorrt_llm.models/*` path.

## 1. Identify Which RTX 6000 You Have

Run:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

Use the result to choose `CUDA_ARCHS`:

- `RTX 6000 Ada` or `NVIDIA RTX 6000 Ada Generation`
  - use `CUDA_ARCHS="89-real"`
  - this is the closest match to the environment already tested
- `Quadro RTX 6000` or the older Turing-based `RTX 6000`
  - use `CUDA_ARCHS="75-real"`
  - start with the PyTorch backend only
  - the current AutoDeploy scripts request `bfloat16`, so they may need an
    `fp16` adjustment on this older GPU

## 2. Clone the Patched Branch

```bash
git clone git@github.com:masahiro-kubota/TensorRT-LLM.git
cd TensorRT-LLM
git checkout codex/qwen35-wheel-build-fixes
```

## 3. Build a Local TRT-LLM Development Image

Build the image with the same tag expected by the helper scripts:

```bash
make -C docker devel_build \
  IMAGE_WITH_TAG="local/trtllm-qwen35-main-runtime:25.06" \
  CUDA_ARCHS="89-real"
```

If your GPU is the older Turing-based RTX 6000, replace `89-real` with
`75-real`.

## 4. Build TensorRT-LLM From Source on This Branch

Start the development container:

```bash
make -C docker devel_run \
  IMAGE_WITH_TAG="local/trtllm-qwen35-main-runtime:25.06" \
  LOCAL_USER=1
```

Inside the container, build the wheel and install the repo editable:

```bash
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
python3 scripts/build_wheel.py --cuda_architectures "89-real" --benchmarks
pip install -e .
```

Again, replace `89-real` with `75-real` on the older Turing card.

Why this step matters:

- the helper scripts mount the repo into the runtime container
- they expect the built libraries under the source tree, especially
  `tensorrt_llm/libs`
- clone alone does not create those binaries

## 5. Provide the Local Checkpoint Path

The scripts expect a local HF checkpoint for `Qwen3.5-0.8B`.

Set:

```bash
export QWEN35_HOST_MODEL_DIR=/absolute/path/to/Qwen3.5-0.8B
export TRTLLM_QWEN35_RUNTIME_IMAGE=local/trtllm-qwen35-main-runtime:25.06
```

The mounted checkpoint should contain the original VLM-style HF config. The
helper scripts will derive the text-only mirror locally.

## 6. First Validation: PyTorch Backend

Run the smallest validation first:

```bash
bash examples/llm-api/qwen35_local_runtime/scripts/run_qwen35_pytorch_backend_probe.sh
```

If that works, the normalized text mirror and `_torch` runtime path are both in
good shape.

Then run the long-context measurement:

```bash
bash examples/llm-api/qwen35_local_runtime/scripts/run_qwen35_pytorch_backend_longctx_perf.sh
```

This is the script that produced the reference result note in
`QWEN35_PYTORCH_BACKEND_LONGCTX_RESULTS.md`.

## 7. Optional Validation: AutoDeploy

The AutoDeploy text-only probe is:

```bash
bash examples/llm-api/qwen35_local_runtime/scripts/run_qwen35_autodeploy_trtllm_probe.sh
```

The current checked-in AutoDeploy probe uses:

- `runtime="trtllm"`
- `compile_backend="torch-simple"`
- `attn_backend="torch"`
- `torch_dtype="bfloat16"`

That setup was chosen to maximize bring-up stability, not speed.

On an RTX 6000 Ada, it is reasonable to try this as-is.

On the older Turing-based RTX 6000:

- do not assume the AutoDeploy scripts will work unchanged
- start with the PyTorch backend path first
- if AutoDeploy is required, expect to adjust the dtype from `bfloat16` to
  `float16`

## 8. Expected Artifacts

The scripts will create local artifacts under:

- `examples/llm-api/qwen35_local_runtime/artifacts/`
- `examples/llm-api/qwen35_local_runtime/logs/`

These outputs are intentionally not committed to the repo.

## 9. Common Failure Modes

### `tensorrt_llm/libs` is missing

Cause:

- `scripts/build_wheel.py` was not run on this clone

Fix:

- rerun the source build step inside the devel container

### Import errors during simple probes

Cause:

- wrong branch or incomplete source build

Fix:

- confirm the branch is `codex/qwen35-wheel-build-fixes`
- rebuild the wheel on top of that branch

### AutoDeploy fails on an older RTX 6000

Cause:

- the current checked-in AutoDeploy probe uses `bfloat16`

Fix:

- validate the PyTorch backend first
- then adjust the AutoDeploy probe to `float16` if needed

### `build_qwen35_tiny.sh` fails

Cause:

- this is the known unresolved gap in the classic TensorRT engine path

Fix:

- none yet on this branch
- use `backend=pytorch` or the text-only AutoDeploy probe instead

## Recommended Bring-up Order

Use this order on a fresh RTX 6000 machine:

1. clone the branch
2. build the devel image
3. run `scripts/build_wheel.py`
4. `pip install -e .`
5. run `run_qwen35_pytorch_backend_probe.sh`
6. run `run_qwen35_pytorch_backend_longctx_perf.sh`
7. only then try AutoDeploy

That keeps the first success criterion narrow and makes it much easier to tell
whether the remaining issue is model support, dtype, or runtime integration.
