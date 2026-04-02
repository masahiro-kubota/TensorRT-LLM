#!/usr/bin/env python3
"""Minimal TRT-LLM PyTorch-backend probe for local Qwen3.5-0.8B text mirror."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tensorrt_llm.llmapi import LLM
from tensorrt_llm.sampling_params import SamplingParams


MODEL_DIR = Path(
    "/workspace/runtime/artifacts/Qwen3.5-0.8B-text-mirror"
)
PROMPTS = [
    "Say hello in 5 words.",
    "Count from one to five.",
]


def main() -> int:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing normalized mirror: {MODEL_DIR}")

    llm = LLM(
        model=str(MODEL_DIR),
        backend="pytorch",
        trust_remote_code=True,
        max_batch_size=1,
        max_seq_len=64,
        max_num_tokens=16,
    )
    try:
        outputs = llm.generate(
            PROMPTS,
            sampling_params=SamplingParams(max_tokens=8),
            use_tqdm=False,
        )
        payload = []
        for prompt, output in zip(PROMPTS, outputs, strict=True):
            payload.append(
                {
                    "prompt": prompt,
                    "text": output.outputs[0].text,
                    "token_ids": output.outputs[0].token_ids,
                }
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
