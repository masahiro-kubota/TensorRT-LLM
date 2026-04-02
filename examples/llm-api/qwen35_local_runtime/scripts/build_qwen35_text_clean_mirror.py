#!/usr/bin/env python3
"""Build a clean text-only mirror from the local Qwen3.5-0.8B VLM checkpoint.

The local checkpoint is a VLM and stores text weights under
``model.language_model.*`` plus vision / MTP weights in the same safetensors
file. AutoDeploy HF loading is much happier if we hand it a plain text-only
checkpoint whose state dict matches the dense text model directly.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import safetensors.torch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]

SRC = Path(
    os.environ.get(
        "QWEN35_VLM_SRC_DIR",
        "/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B",
    )
)
DST = Path(
    os.environ.get(
        "QWEN35_TEXT_CLEAN_DST_DIR",
        str(RUNTIME_ROOT / "artifacts" / "Qwen3.5-0.8B-text-clean"),
    )
)

VLM_ARCHS = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def normalize_qwen35_text_config(config_dict: dict) -> dict:
    architectures = config_dict.get("architectures") or []
    if architectures and architectures[0] in VLM_ARCHS:
        text_config = dict(config_dict.get("text_config") or {})
    else:
        text_config = dict(config_dict)
    if not text_config:
        raise ValueError("Qwen3.5 config is missing usable text_config")

    if "quantization_config" not in text_config and "quantization_config" in config_dict:
        text_config["quantization_config"] = dict(config_dict["quantization_config"])

    rope_parameters = dict(text_config.pop("rope_parameters", {}) or {})
    rope_scaling = dict(text_config.get("rope_scaling") or {})
    if rope_parameters:
        rope_theta = rope_parameters.pop("rope_theta", None)
        if rope_theta is not None:
            text_config.setdefault("rope_theta", rope_theta)
        partial_rotary_factor = rope_parameters.pop("partial_rotary_factor", None)
        if partial_rotary_factor is not None:
            text_config.setdefault("partial_rotary_factor", partial_rotary_factor)
        if rope_parameters:
            rope_scaling = rope_parameters | rope_scaling
    if rope_scaling:
        has_mrope = "mrope_section" in rope_scaling or rope_scaling.get("mrope_interleaved", False)
        if has_mrope:
            rope_scaling["type"] = "mrope"
            rope_scaling.pop("rope_type", None)
        elif "type" not in rope_scaling and "rope_type" in rope_scaling:
            rope_scaling["type"] = rope_scaling.pop("rope_type")
        text_config["rope_scaling"] = rope_scaling

    text_config["architectures"] = ["Qwen3_5ForCausalLM"]
    text_config["model_type"] = "qwen3_5_text"
    text_config.setdefault("num_experts", 0)
    text_config.setdefault("num_experts_per_tok", 0)
    text_config.setdefault("moe_intermediate_size", 0)
    text_config.setdefault("shared_expert_intermediate_size", 0)
    return text_config


def copy_tokenizer_files(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.name.startswith("."):
            continue
        if item.name.endswith(".safetensors") or item.name.endswith(".safetensors.index.json"):
            continue
        if item.name == "config.json":
            continue
        target = dst / item.name
        if item.is_file():
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)


def rewrite_weights(src: Path, dst: Path) -> None:
    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        shard_state = safetensors.torch.load_file(str(shard))
        for key, value in shard_state.items():
            if key.startswith("model.language_model."):
                text_key = key.removeprefix("model.language_model.")
                if text_key == "lm_head.weight":
                    state[text_key] = value
                else:
                    state[f"model.{text_key}"] = value
            elif key == "lm_head.weight":
                state[key] = value

    if not state:
        raise RuntimeError("No language-model weights were extracted from source checkpoint")

    out_name = "model.safetensors"
    safetensors.torch.save_file(state, str(dst / out_name))
    weight_map = {key: out_name for key in sorted(state)}
    index_payload = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index_payload, indent=2) + "\n")


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    raw_config = json.loads((SRC / "config.json").read_text())
    normalized = normalize_qwen35_text_config(raw_config)
    (DST / "config.json").write_text(json.dumps(normalized, indent=2) + "\n")

    copy_tokenizer_files(SRC, DST)
    rewrite_weights(SRC, DST)
    print(DST)


if __name__ == "__main__":
    main()
