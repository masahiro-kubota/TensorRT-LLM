#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[1]

VLM_ARCHS = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def normalize_qwen35_config(config_dict: dict) -> dict:
    architectures = config_dict.get("architectures") or []
    if architectures and architectures[0] in VLM_ARCHS:
        text_config = dict(config_dict.get("text_config") or {})
    else:
        text_config = dict(config_dict)
    if not text_config:
        raise ValueError("Qwen3.5 config is missing a usable text_config")

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
        has_mrope = ("mrope_section" in rope_scaling or rope_scaling.get("mrope_interleaved", False))
        if has_mrope:
            rope_scaling["type"] = "mrope"
            rope_scaling.pop("rope_type", None)
        elif "type" not in rope_scaling and "rope_type" in rope_scaling:
            rope_scaling["type"] = rope_scaling.pop("rope_type")
        text_config["rope_scaling"] = rope_scaling

    is_moe = "num_experts" in text_config and text_config["num_experts"] > 0
    if is_moe:
        text_config["architectures"] = ["Qwen3_5MoeForCausalLM"]
    else:
        text_config["architectures"] = ["Qwen3_5ForCausalLM"]
        text_config.setdefault("num_experts", 0)
        text_config.setdefault("num_experts_per_tok", 0)
        text_config.setdefault("moe_intermediate_size", 0)
        text_config.setdefault("shared_expert_intermediate_size", 0)

    if text_config.get("model_type") == "qwen3_5":
        text_config["model_type"] = "qwen3_5_text"

    return text_config


def main() -> None:
    src = Path(
        os.environ.get(
            "QWEN35_VLM_SRC_DIR",
            "/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B",
        )
    )
    dst = Path(
        os.environ.get(
            "QWEN35_TEXT_MIRROR_DST_DIR",
            str(RUNTIME_ROOT / "artifacts" / "Qwen3.5-0.8B-text-mirror"),
        )
    )
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    raw_config = json.loads((src / "config.json").read_text())
    normalized = normalize_qwen35_config(raw_config)
    (dst / "config.json").write_text(json.dumps(normalized, indent=2) + "\n")

    for item in src.iterdir():
        if item.name == "config.json":
            continue
        if item.name.startswith("."):
            continue
        target = dst / item.name
        if item.is_file():
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)

    print(dst)


if __name__ == "__main__":
    main()
