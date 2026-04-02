# Qwen3.5 Local Source Fixes

This note explains the local source patch set preserved in commit `454157c`
(`Preserve local qwen35 wheel/runtime fixes`).

It is intentionally focused on two questions:

1. Why was each source change needed?
2. How was the code changed to unblock the local wheel/runtime flow?

This document is about the source-level fixes only. The later reproducibility
scripts added under `examples/llm-api/qwen35_local_runtime/` are separate.

## Goal

The local goal was not to add full upstream-ready Qwen3.5 support in one shot.
It was narrower:

- keep TRT-LLM main/source buildable in the local container
- make import-time failures less fatal in a mixed source/container environment
- let TRT-LLM discover a normalized `Qwen3.5-0.8B` text backbone config
- make `backend=pytorch` and AutoDeploy text-only probes runnable locally

## Non-goals

This patch set does not complete the classic TensorRT engine path for dense
Qwen3.5. In particular, it does not add dense `Qwen3_5ForCausalLM` support to
`tensorrt_llm.models.MODEL_MAP` or a full converter for the classic
`trtllm-bench build` path.

## Change Groups

### 1. Wheel build and import-time compatibility

Files:

- `scripts/build_wheel.py`
- `tensorrt_llm/__init__.py`
- `tensorrt_llm/_torch/__init__.py`
- `tensorrt_llm/llmapi/__init__.py`
- `tensorrt_llm/commands/bench.py`
- `tensorrt_llm/_torch/models/__init__.py`
- `tensorrt_llm/_torch/auto_deploy/transform/library/__init__.py`

Why:

- The local environment mixed TRT-LLM main/source with an older base container.
- Some optional modules pulled in unavailable dependencies at import time and
  blocked simple probes before model loading even started.
- The wheel-build flow also needed a way to skip reinstalling requirements that
  were already satisfied in the prepared environment.

How:

- `scripts/build_wheel.py`
  - Added `TRTLLM_SKIP_REQUIREMENTS_INSTALL=1` so the local wheel build can
    reuse the prepared environment without forcing another `pip install -r`.
- `tensorrt_llm/__init__.py`
  - Wrapped broad top-level imports and `_init()` in `try/except`.
  - When an optional path fails, the package now exposes `None` placeholders and
    prints a probe-oriented diagnostic instead of failing immediately.
- `tensorrt_llm/_torch/__init__.py`
  - Made `_torch.LLM` import optional for the same reason.
- `tensorrt_llm/llmapi/__init__.py`
  - Made `visual_gen` optional and only exported it when import succeeds.
- `tensorrt_llm/commands/bench.py`
  - Registered `visual_gen_command` only when its import succeeds, so
    `trtllm-bench` still starts in the local container.
- `tensorrt_llm/_torch/models/__init__.py`
  - Relaxed Mistral imports so unrelated optional model dependencies do not
    prevent Qwen3.5 probing.
- `tensorrt_llm/_torch/auto_deploy/transform/library/__init__.py`
  - Switched transform-module auto-import to a warning-and-skip path on
    `ModuleNotFoundError`, which keeps AutoDeploy usable even when a transform's
    optional dependency is absent.

Net effect:

- The package becomes tolerant enough to load the parts needed for local Qwen3.5
  experiments, instead of dying during unrelated optional imports.

### 2. Qwen3.5 config discovery fallback

Files:

- `tensorrt_llm/llmapi/llm_args.py`
- `tensorrt_llm/models/automodel.py`

Why:

- The local checkpoint started as a VLM checkpoint with nested Qwen3.5 text
  config information.
- `transformers.AutoConfig.from_pretrained(...)` was not sufficient for the
  normalized `qwen3_5_text` / `qwen3_5_moe_text` path in this environment.
- Without a fallback, TRT-LLM failed very early while merely trying to identify
  the model format or infer the TRT-LLM model class.

How:

- `tensorrt_llm/llmapi/llm_args.py`
  - In `get_model_format`, kept the normal HF config path first.
  - If that path fails and the model type is one of the local Qwen3.5 variants,
    it now falls back to `_torch.pyexecutor.config_utils.load_pretrained_config`.
- `tensorrt_llm/models/automodel.py`
  - Added a shared `_load_hf_config(...)` helper.
  - It tries HF `AutoConfig` first, then uses
    `load_pretrained_config(...)` and accepts the result only for
    `qwen3_5_text` / `qwen3_5_moe_text`.
  - Both `AutoConfig.from_hugging_face(...)` and
    `AutoModelForCausalLM.get_trtllm_model_class(...)` now use this helper.

Net effect:

- TRT-LLM can identify the normalized Qwen3.5 text checkpoint instead of
  rejecting it during config loading.

### 3. Dense Qwen3.5 text model for AutoDeploy

Files:

- `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5.py`
- `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py`
- `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`

Why:

- Upstream source already had Qwen3.5 MoE-oriented building blocks, but the
  local text-only `0.8B` probe needed the dense channel-mixer path.
- The checkpoint used dense MLP weights, while some downstream code expected the
  MoE-style nested naming pattern.
- Without a dense custom model class plus compatible weight-name handling,
  AutoDeploy could not load the normalized text backbone.

How:

- `modeling_qwen3_5.py`
  - Added a minimal dense `Qwen3_5TextConfig`.
  - Reused the existing Qwen3.5 hybrid token-mixer blocks from the MoE
    implementation:
    - full attention
    - GatedDeltaNet linear attention
    - mRoPE / RMSNorm helpers
  - Replaced the routed MoE channel mixer with a dense SwiGLU MLP.
  - Added:
    - `Qwen3_5DecoderLayer`
    - `Qwen3_5TextModel`
    - `Qwen3_5ForCausalLM`
  - Registered the config with HF `AutoConfig` and the model class with
    `AutoModelForCausalLMFactory`.
- `custom/__init__.py`
  - Exported `Qwen3_5ForCausalLM`.
- `qwen3_5_weight_mapper.py`
  - Added a dense-MLP regex and mirrored
    `model.layers.N.mlp.{gate,up,down}_proj.*` into
    `model.layers.N.mlp.mlp.{...}` aliases.
  - This keeps dense checkpoints compatible with loader logic that still expects
    the nested `.mlp.` naming.

Net effect:

- AutoDeploy can instantiate and load a dense Qwen3.5 text backbone for local
  text-only probes.

### 4. AutoDeploy runtime glue fixes

Files:

- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py`
- `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py`
- `tensorrt_llm/_torch/distributed/communicator.py`

Why:

- The local runtime path hit a few integration mismatches that were not
  fundamental model issues:
  - config/backend names did not line up exactly
  - executor startup assumed a distributed state that was not initialized yet
  - some TorchDist code assumed Ray was available

How:

- `attention_interface.py`
  - Added an alias from `MultiHeadLatentAttention` to `torch_mla`.
  - This lets config files refer to the generic MLA name while still resolving
    to a concrete registered backend.
- `ad_executor.py`
  - Changed port selection so multi-rank startup uses `mpi_broadcast(...)`
    before the higher-level `Distributed` wrapper exists.
  - Single-rank startup now just picks a local free port directly.
- `communicator.py`
  - Added a non-Ray fallback for `TorchDist`.
  - When Ray is not initialized, it now uses the current host IP and current
    CUDA device instead of aborting.

Net effect:

- AutoDeploy executor startup becomes usable in the local single-node,
  non-Ray experiment flow.

### 5. AutoDeploy config pruning for the local container

Files:

- `tensorrt_llm/_torch/auto_deploy/config/default.yaml`
- `tensorrt_llm/_torch/auto_deploy/config/transformers.yaml`

Why:

- The source tree was newer than the local container/runtime environment.
- Some transform entries or expectations in the default config were either not
  available locally, too aggressive for the probe, or caused failures unrelated
  to the actual Qwen3.5 text-model bring-up.
- The immediate goal was a stable local execution path, not a maximal transform
  set.

How:

- Removed or disabled several transform entries that were not needed for the
  local Qwen3.5 text-only path.
- Simplified sharding behavior and set `shard_all_unprocessed: true`.
- Switched `insert_cached_mla_attention.backend` to
  `MultiHeadLatentAttention`, which is then resolved through the new alias.
- Set `resize_kv_cache.free_mem_ratio: 0.0`.
- Dropped several `expect_mem_change` and related assumptions that were too
  strict for this local environment.
- Kept the compile step on `torch-compile` in config, while later probe scripts
  were still free to override this with safer runtime flags.

Net effect:

- The default AutoDeploy config becomes conservative enough to run the local
  Qwen3.5 probes instead of failing in transform/config plumbing.

### 6. PyExecutor timing dependency cleanup

Files:

- `tensorrt_llm/_torch/pyexecutor/time_utils.py`
- `tensorrt_llm/_torch/pyexecutor/perf_metrics_manager.py`
- `tensorrt_llm/_torch/pyexecutor/py_executor.py`

Why:

- The local `_torch` execution path still needed steady-clock timestamps, but
  importing them through `tensorrt_llm.serve.responses_utils` added an
  unnecessary dependency edge into the serve stack.

How:

- Added `_torch/pyexecutor/time_utils.py` with a thin wrapper around
  `tensorrt_llm.bindings.steady_clock_now`.
- Updated the pyexecutor timing users to import from this local helper instead
  of the serve package.

Net effect:

- The `_torch` execution path depends on a smaller and more direct timing
  utility, which reduces import coupling for local probes.

### 7. C++ portability fixes for local wheel build

Files:

- `cpp/tensorrt_llm/kernels/indexerTopK.cu`
- `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h`
- `cpp/tensorrt_llm/thop/allreduceOp.cpp`

Why:

- The local build environment surfaced missing C/C++ definitions and NCCL
  feature mismatches.
- In particular, the code path for NCCL symmetric/window-buffer allreduce
  assumed NCCL support that was not available in the local compile target.

How:

- Added `<cfloat>` includes where needed for compile-time constants.
- In `allreduceOp.cpp`
  - introduced `TRTLLM_THOP_HAS_NCCL_WINDOW`
  - guarded the NCCL window-buffer path on NCCL version support
  - fell back to plain NCCL allreduce when that feature is unavailable
  - turned preallocation into a no-op in unsupported builds instead of failing

Net effect:

- The local wheel build becomes tolerant of older or differently configured NCCL
  environments.

## Outcome

With this patch set, the local environment was able to:

- build and preserve a usable TRT-LLM main/source wheel/runtime environment
- normalize the local `Qwen3.5-0.8B` VLM checkpoint into a text-only mirror
- run `backend=pytorch` locally on the normalized text mirror
- run AutoDeploy text-only probes locally on the normalized text mirror

What remains blocked is the classic TensorRT engine path for dense Qwen3.5 in
`tensorrt_llm.models/*`.
