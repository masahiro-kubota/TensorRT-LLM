#!/usr/bin/env python3
"""Warm latency probe for TRT-LLM PyTorch backend with Qwen3.5-0.8B."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from tensorrt_llm.llmapi import LLM
from tensorrt_llm.sampling_params import SamplingParams

MODEL_DIR = Path("/workspace/runtime/artifacts/Qwen3.5-0.8B-text-mirror")
PROMPT = "Say hello in 5 words."
SAMPLING = SamplingParams(max_tokens=8)


def run_once(llm: LLM) -> dict:
    started = time.perf_counter()
    outputs = llm.generate(PROMPT, sampling_params=SAMPLING, use_tqdm=False)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": elapsed,
        "text": outputs.outputs[0].text,
        "token_ids": outputs.outputs[0].token_ids,
    }


def main() -> int:
    llm = LLM(
        model=str(MODEL_DIR),
        backend="pytorch",
        trust_remote_code=True,
        max_batch_size=1,
        max_seq_len=64,
        max_num_tokens=16,
    )
    try:
        warmup = run_once(llm)
        runs = [run_once(llm) for _ in range(3)]
    finally:
        llm.shutdown()

    times = [run["elapsed_s"] for run in runs]
    payload = {
        "prompt": PROMPT,
        "max_seq_len": 64,
        "max_num_tokens": 16,
        "max_tokens": 8,
        "warmup_s": warmup["elapsed_s"],
        "mean_s": statistics.fmean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "runs": runs,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
