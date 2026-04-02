#!/usr/bin/env python3
"""Small AutoDeploy runtime=trtllm probe for dense Qwen3.5 text mirror."""

from __future__ import annotations

import json
from pathlib import Path

from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.sampling_params import SamplingParams


MODEL_DIR = Path("/workspace/runtime/artifacts/Qwen3.5-0.8B-text-clean")
PROMPTS = [
    {"prompt": "Say hello in 5 words."},
]


def main() -> int:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing clean text mirror: {MODEL_DIR}")

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
        compile_backend="torch-simple",
        attn_backend="torch",
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    try:
        outs = llm.generate(
            PROMPTS,
            sampling_params=SamplingParams(max_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
            use_tqdm=False,
        )
        payload = []
        for prompt, output in zip(PROMPTS, outs, strict=True):
            payload.append(
                {
                    "prompt": prompt["prompt"],
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
