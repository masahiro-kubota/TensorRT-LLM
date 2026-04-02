#!/usr/bin/env python3
"""Measure Qwen3.5-0.8B PyTorch backend performance at input=2048, output=40."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from tensorrt_llm.llmapi import LLM
from tensorrt_llm.metrics.enums import MetricNames
from tensorrt_llm.sampling_params import SamplingParams

MODEL_DIR = Path("/workspace/runtime/artifacts/Qwen3.5-0.8B-text-mirror")
INPUT_TOKENS = 2048
OUTPUT_TOKENS = 40
RUNS = 3
WARMUP_INPUT_TOKENS = 128


def _make_prompt_ids(llm: LLM, *, length: int, seed: int) -> list[int]:
    tokenizer = llm.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required to build fixed-length prompts.")

    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    vocab_size = getattr(hf_tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("Could not determine tokenizer vocab size.")

    special_ids = set(getattr(hf_tokenizer, "all_special_ids", []) or [])
    rng = random.Random(seed)
    prompt_ids: list[int] = []
    while len(prompt_ids) < length:
        token_id = rng.randrange(vocab_size)
        if token_id in special_ids:
            continue
        prompt_ids.append(token_id)
    return prompt_ids


def _metrics_to_dict(metrics: dict) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        if hasattr(key, "value"):
            payload[key.value] = value
        else:
            payload[str(key)] = value
    return payload


def _run_once(llm: LLM, *, prompt_ids: list[int], cache_salt: str) -> dict:
    sampling = SamplingParams(
        max_tokens=OUTPUT_TOKENS,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        return_perf_metrics=True,
    )
    output = llm.generate(
        prompt_ids,
        sampling_params=sampling,
        use_tqdm=False,
        cache_salt=cache_salt,
    )

    metrics = _metrics_to_dict(output.metrics_dict)
    ttft_s = metrics.get(MetricNames.TTFT.value)
    tpot_s = metrics.get(MetricNames.TPOT.value)
    e2e_s = metrics.get(MetricNames.E2E.value)

    record = {
        "prompt_tokens": len(prompt_ids),
        "output_tokens": output.outputs[0].length,
        "finish_reason": output.outputs[0].finish_reason,
        "metrics": metrics,
        "text_preview": output.outputs[0].text[:120],
    }

    if ttft_s and ttft_s > 0:
        record["prefill_tok_s_wall"] = len(prompt_ids) / ttft_s
    if tpot_s and tpot_s > 0:
        record["decode_tok_s_wall"] = 1.0 / tpot_s
    if e2e_s and e2e_s > 0:
        record["overall_tok_s_wall"] = (len(prompt_ids) + output.outputs[0].length) / e2e_s

    time_breakdown = output.time_breakdown_metrics or {}
    record["time_breakdown_metrics"] = time_breakdown

    ctx_gpu_forward_ms = time_breakdown.get("ctx_gpu_forward_time")
    if ctx_gpu_forward_ms and ctx_gpu_forward_ms > 0:
        record["prefill_tok_s_gpu_forward"] = len(prompt_ids) / (ctx_gpu_forward_ms / 1000.0)

    step_metrics = time_breakdown.get("step_metrics") or []
    total_decode_gpu_forward_ms = sum(step.get("gpu_forward_time", 0.0) for step in step_metrics)
    if total_decode_gpu_forward_ms > 0:
        record["decode_tok_s_gpu_forward"] = len(step_metrics) / (total_decode_gpu_forward_ms / 1000.0)

    return record


def main() -> int:
    llm = LLM(
        model=str(MODEL_DIR),
        backend="pytorch",
        trust_remote_code=True,
        max_batch_size=1,
        max_seq_len=INPUT_TOKENS + OUTPUT_TOKENS,
        max_num_tokens=INPUT_TOKENS + OUTPUT_TOKENS,
        kv_cache_config={"enable_block_reuse": False},
    )
    try:
        warmup_prompt_ids = _make_prompt_ids(llm, length=WARMUP_INPUT_TOKENS, seed=0)
        llm.generate(
            warmup_prompt_ids,
            sampling_params=SamplingParams(max_tokens=1, top_k=1, top_p=1.0, temperature=1.0),
            use_tqdm=False,
            cache_salt="warmup",
        )

        runs = []
        for idx in range(RUNS):
            prompt_ids = _make_prompt_ids(llm, length=INPUT_TOKENS, seed=100 + idx)
            runs.append(_run_once(llm, prompt_ids=prompt_ids, cache_salt=f"run-{idx}"))
    finally:
        llm.shutdown()

    prefill_wall = [run["prefill_tok_s_wall"] for run in runs if "prefill_tok_s_wall" in run]
    decode_wall = [run["decode_tok_s_wall"] for run in runs if "decode_tok_s_wall" in run]
    prefill_gpu = [run["prefill_tok_s_gpu_forward"] for run in runs if "prefill_tok_s_gpu_forward" in run]
    decode_gpu = [run["decode_tok_s_gpu_forward"] for run in runs if "decode_tok_s_gpu_forward" in run]

    payload = {
        "model_dir": str(MODEL_DIR),
        "backend": "pytorch",
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "max_seq_len": INPUT_TOKENS + OUTPUT_TOKENS,
        "max_num_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
        "runs": runs,
        "summary": {
            "prefill_tok_s_wall_mean": statistics.fmean(prefill_wall) if prefill_wall else None,
            "prefill_tok_s_wall_median": statistics.median(prefill_wall) if prefill_wall else None,
            "decode_tok_s_wall_mean": statistics.fmean(decode_wall) if decode_wall else None,
            "decode_tok_s_wall_median": statistics.median(decode_wall) if decode_wall else None,
            "prefill_tok_s_gpu_forward_mean": statistics.fmean(prefill_gpu) if prefill_gpu else None,
            "decode_tok_s_gpu_forward_mean": statistics.fmean(decode_gpu) if decode_gpu else None,
            "ttft_s_mean": statistics.fmean(
                run["metrics"][MetricNames.TTFT.value] for run in runs if MetricNames.TTFT.value in run["metrics"]
            ),
            "tpot_s_mean": statistics.fmean(
                run["metrics"][MetricNames.TPOT.value] for run in runs if MetricNames.TPOT.value in run["metrics"]
            ),
            "e2e_s_mean": statistics.fmean(
                run["metrics"][MetricNames.E2E.value] for run in runs if MetricNames.E2E.value in run["metrics"]
            ),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
