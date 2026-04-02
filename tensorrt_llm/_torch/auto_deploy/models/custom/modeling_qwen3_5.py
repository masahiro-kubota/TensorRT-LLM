# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dense Qwen3.5 text model for AutoDeploy.

This is the dense counterpart to ``modeling_qwen3_5_moe.py``. It intentionally
reuses the same Qwen3.5 hybrid token-mixer building blocks:
  * full attention layers
  * GatedDeltaNet linear-attention layers
  * mRoPE handling

The only architectural difference is the channel mixer: dense SwiGLU MLP
instead of routed MoE.

This file is text-only on purpose. It is sufficient for validating that
AutoDeploy ``runtime=trtllm`` can ingest the dense Qwen3.5 text backbone carried
inside the local VLM checkpoint once the checkpoint is normalized into a clean
text-only mirror.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig
from transformers.generation import GenerationMixin
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

from .modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeCausalLMOutput,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeOutput,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)


class Qwen3_5TextConfig(PretrainedConfig):
    """Minimal dense Qwen3.5 text config.

    We intentionally mirror the Qwen3.5 text fields carried by the local
    Qwen3.5-0.8B VLM checkpoint, but without any MoE-specific fields.
    """

    model_type = "qwen3_5_text"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        intermediate_size=3584,
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        layer_types=None,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
            }
        self.rope_parameters = rope_parameters

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Dense model: explicitly keep MoE knobs disabled so downstream helpers
        # that inspect these attributes do not misclassify the model.
        self.num_experts = 0
        self.num_experts_per_tok = 0
        self.moe_intermediate_size = 0
        self.shared_expert_intermediate_size = 0

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


try:
    AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig)
    except ValueError:
        pass


class Qwen3_5MLP(Qwen3_5MoeMLP):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config, intermediate_size=config.intermediate_size)


class Qwen3_5DecoderLayer(nn.Module):
    """Single dense Qwen3.5 decoder layer."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5PreTrainedModel(Qwen3_5MoePreTrainedModel):
    config_class = Qwen3_5TextConfig
    _no_split_modules = ["Qwen3_5DecoderLayer"]


class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    """Dense Qwen3.5 text backbone."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        pad_token_id = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5MoeOutput]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if rope_cos is not None and rope_sin is not None:
            position_embeddings = (rope_cos, rope_sin)
        elif position_embeddings is None:
            if position_ids is None:
                seq_len = inputs_embeds.shape[1]
                position_ids = torch.arange(seq_len, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings=position_embeddings)

        hidden_states = self.norm(hidden_states)
        return Qwen3_5MoeOutput(last_hidden_state=hidden_states)


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    """Dense Qwen3.5 causal LM for AutoDeploy."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3_5TextConfig, **kwargs):
        super().__init__(config)
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5MoeCausalLMOutput]:
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return Qwen3_5MoeCausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3_5TextConfig", Qwen3_5ForCausalLM)
