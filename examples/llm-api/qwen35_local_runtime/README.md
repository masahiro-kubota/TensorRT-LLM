# Qwen3.5-0.8B Local Runtime Notes

This directory preserves the local experiment scripts that were used to make a
`Qwen3.5-0.8B` VLM checkpoint usable with TRT-LLM main/source on one machine.

It contains only reproducibility assets:

- helper scripts
- run wrappers
- result notes

It intentionally does not contain large local artifacts such as:

- `artifacts/Qwen3.5-0.8B-text-mirror`
- `artifacts/Qwen3.5-0.8B-text-clean`
- build logs

Those are regenerated locally when needed.

## Status

- Public `tensorrt-llm==1.2.0` wheel/release did not support dense `qwen3_5`
- TRT-LLM main/source does contain `qwen3_5` text-backbone support
- The local checkpoint is a VLM checkpoint:
  `Qwen3_5ForConditionalGeneration` with nested `text_config`
- TRT-LLM main can normalize that VLM checkpoint into a text-backbone config
- Classic TensorRT engine build is still blocked for dense `Qwen3_5ForCausalLM`
  in `tensorrt_llm.models.MODEL_MAP`
- PyTorch backend works locally for the normalized text mirror
- AutoDeploy also runs locally for text-only probes, but the tested settings
  here used `compile_backend=torch-simple` and `attn_backend=torch`
- AutoDeploy long-context measurements were added for `torch`, `flashinfer`,
  and `trtllm` attention backends under `compile_backend=torch-simple`

## Environment

The run scripts assume:

- this repository is mounted into the container as `/workspace/src`
- this example directory is mounted into the container as `/workspace/runtime`

Override these host-side settings with env vars when needed:

- `TRTLLM_QWEN35_RUNTIME_IMAGE`
  - default: `local/trtllm-qwen35-main-runtime:25.06`
- `QWEN35_HOST_MODEL_DIR`
  - default: `/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B`

## Included Files

- `scripts/normalize_qwen35_vlm_checkpoint.py`
- `scripts/build_qwen35_text_clean_mirror.py`
- `scripts/build_qwen35_tiny.sh`
- `scripts/qwen35_pytorch_backend_probe.py`
- `scripts/run_qwen35_pytorch_backend_probe.sh`
- `scripts/qwen35_pytorch_backend_latency.py`
- `scripts/run_qwen35_pytorch_backend_latency.sh`
- `scripts/qwen35_pytorch_backend_longctx_perf.py`
- `scripts/run_qwen35_pytorch_backend_longctx_perf.sh`
- `scripts/qwen35_autodeploy_trtllm_probe.py`
- `scripts/run_qwen35_autodeploy_trtllm_probe.sh`
- `scripts/qwen35_autodeploy_trtllm_latency.py`
- `scripts/run_qwen35_autodeploy_trtllm_latency.sh`
- `scripts/qwen35_autodeploy_trtllm_longctx_perf.py`
- `scripts/run_qwen35_autodeploy_trtllm_longctx_perf.sh`
- `QWEN35_RTX6000_SETUP.md`
- `QWEN35_SOURCE_FIXES.md`
- `QWEN35_PYTORCH_BACKEND_LONGCTX_RESULTS.md`

## Notes

- This is a text-backbone TRT-LLM path. It does not prove dense Qwen3.5 VLM
  end-to-end support in TRT-LLM.
- The TensorRT backend remains blocked by a deeper gap than registration:
  legacy `tensorrt_llm.models/*` still lacks a dense Qwen3.5 hybrid
  linear-attention model plus converter, while `_torch/*` does not.
