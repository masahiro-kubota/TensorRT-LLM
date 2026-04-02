# Draft Issue: Qwen3.5 Dense Support in Legacy TensorRT Engine Backend

## Suggested Title

`[New Model]: Support Qwen3.5 dense model in legacy TensorRT engine backend`

## Recommended Body

```md
### The model to consider.

Qwen3.5 dense text-only model in the legacy TensorRT engine path, i.e. the `tensorrt_llm.models` / `trtllm-bench build` flow that produces a real TensorRT engine.

This issue is specifically **not** about:

- PyTorch backend
- AutoDeploy
- multimodal / VL
- MoE
- MTP

I could not find an existing issue for this exact scope. The nearby Qwen3.5 issues I found are for AutoDeploy / PyTorch backend instead, e.g. #11440, #11674, and #11947.

### The closest model TensorRT-LLM already supports.

- Qwen3 dense in the legacy TensorRT engine path via PR #5650
- Qwen3.5 support in newer execution paths such as AutoDeploy / `_torch` via PRs like #11394, #12242, and #12302

### What's your difficulty of supporting the model you want?

With a normalized Qwen3.5 dense text-only checkpoint, the build flow can reach the TensorRT backend entry, but support is still incomplete in the legacy model stack.

Observed gaps include:

- missing or incomplete architecture routing for Qwen3.5 in the legacy model registry
- missing legacy config handling for Qwen3.5-specific model types
- no end-to-end legacy implementation for the hybrid full-attention + linear-attention stack
- missing legacy-side weight conversion / packing for Qwen3.5-specific HF weights

In other words, Qwen3.5 is partially supported upstream in newer execution paths, but not yet end-to-end in the legacy TensorRT engine backend.

Proposed initial scope:

- dense only
- text-only checkpoint
- single GPU
- BF16 or FP16
- minimal `convert + build + generate`

Non-goals for the initial support:

- MoE
- multimodal / VL
- FP8 / NVFP4
- long-context optimization
- multi-node / advanced parallelism

Expected implementation surface:

- `tensorrt_llm/models/__init__.py`
- `tensorrt_llm/models/automodel.py`
- `tensorrt_llm/models/qwen/config.py`
- `tensorrt_llm/models/qwen/model.py`
- `tensorrt_llm/models/qwen/convert.py`
- possibly `tensorrt_llm/models/model_weights_loader.py`
- plus any necessary legacy-side linear-attention / hybrid-layer modules

The closest structural precedent seems to be PR #5650, which added Qwen3 dense support in the legacy TensorRT engine path.

Would maintainers accept a focused PR with this scope:

- legacy TensorRT engine backend only
- Qwen3.5 dense only
- minimal support for `convert + build + generate`
- follow-up PRs later for quantization, MoE, and multimodal support

If yes, I can prepare a PR that follows the same reviewable shape as the earlier Qwen3 dense onboarding work.
```

## Notes

- TRT-LLM の `[New Model]` issue テンプレートに寄せた短い版です
- 既存 issue と重ならない点は `legacy TensorRT engine backend` を明示して切っています
- maintainer に最初に聞きたいのは「このスコープの PR を受けるか」です
