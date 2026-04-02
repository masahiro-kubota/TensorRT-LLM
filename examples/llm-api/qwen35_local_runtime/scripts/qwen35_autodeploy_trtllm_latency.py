#!/usr/bin/env python3
"""Warm latency probe for TRT-LLM AutoDeploy with Qwen3.5-0.8B text mirror."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.sampling_params import SamplingParams

MODEL_DIR = Path("/workspace/runtime/artifacts/Qwen3.5-0.8B-text-clean")
PROMPT = {"prompt": "Say hello in 5 words."}
SAMPLING = SamplingParams(max_tokens=8, top_k=1, top_p=1.0, temperature=1.0)
COMPILE_BACKEND = os.environ.get("QWEN35_COMPILE_BACKEND", "torch-simple")
ATTN_BACKEND = os.environ.get("QWEN35_ATTN_BACKEND", "torch")
WARM_RUNS = int(os.environ.get("QWEN35_WARM_RUNS", "5"))


def run_once(llm: LLM) -> dict:
    started = time.perf_counter()
    outputs = llm.generate([PROMPT], sampling_params=SAMPLING, use_tqdm=False)
    elapsed = time.perf_counter() - started
    output = outputs[0].outputs[0]
    return {
        "elapsed_s": elapsed,
        "text": output.text,
        "token_ids": output.token_ids,
    }


def main() -> int:
    payload = {
        "model_dir": str(MODEL_DIR),
        "prompt": PROMPT["prompt"],
        "max_seq_len": 64,
        "max_num_tokens": 16,
        "max_tokens": 8,
        "world_size": 1,
        "runtime": "trtllm",
        "compile_backend": COMPILE_BACKEND,
        "attn_backend": ATTN_BACKEND,
        "dtype": "bfloat16",
    }

    init_started = time.perf_counter()
    llm = LLM(
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        world_size=1,
        runtime="trtllm",
        model_factory="AutoModelForCausalLM",
        trust_remote_code=True,
        max_batch_size=1,
        max_seq_len=64,
        max_num_tokens=16,
        compile_backend=COMPILE_BACKEND,
        attn_backend=ATTN_BACKEND,
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    payload["init_s"] = time.perf_counter() - init_started

    try:
        warmup = run_once(llm)
        runs = [run_once(llm) for _ in range(WARM_RUNS)]
    finally:
        llm.shutdown()

    times = [run["elapsed_s"] for run in runs]
    payload.update(
        {
            "warmup_s": warmup["elapsed_s"],
            "mean_s": statistics.fmean(times),
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
            "runs": runs,
        }
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
