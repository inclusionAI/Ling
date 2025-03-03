# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from dataclasses import dataclass
import copy

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorEmbedding,
    TensorParallelEmbedding,
    TensorReplicatedLinear,
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.models.deepseekv2.position_embedding_deepseekv2 import \
    PositionRotaryEmbedding, DeepseekV2YarnRotaryEmbedding
from atb_llm.models.deepseek.modeling_deepseek import \
    DeepseekMLP, DeepseekMoE
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode

_ROPE_SCALING_KEYS = ["original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"]


@dataclass
class EpSplitParam:
    gatherd_idxs: Any = None
    input_split_sizes: Any = None
    output_splits: Any = None


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class DeepseekV2RMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=weights.dtype)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class DeepseekV2RMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = DeepseekV2RMSNorm(prefix, weights, eps)
        self.anti = DeepseekV2RMSNormBias(f'{prefix}.module', weights, eps)


class DeepseekV2RMSNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = DeepseekV2RMSNorm(f'{prefix}.ori', weights, eps)
        self.anti = DeepseekV2RMSNormBias(f'{prefix}.anti', weights, eps)


class DeepseekV2MLP(DeepseekMLP):
    def __init__(self, prefix, config, weights, intermediate_size=None):
        super().__init__(prefix, config, weights, intermediate_size=None)
        self.act_fn = ACT2FN[config.hidden_act]


class DeepseekV2MoE(DeepseekMoE):
    def __init__(self, prefix, config, weights, shared_mlp_cls):
        super().__init__(prefix, config, weights, shared_mlp_cls)
        self.pack_type = self.shared_experts.pack_type


class FlashDeepseekV2Attention(nn.Module):
    def __init__(self,
                 prefix: str,
                 config,
                 weights):
        super().__init__()
        self.config = config
        self.config = copy.deepcopy(config)
        if hasattr(self.config, 'mla_quantize'):
            self.config.quantize = self.config.mla_quantize
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.is_causal = True
        linear_names = []

        if self.q_lora_rank is None:
            self.q_proj = TensorParallelColumnLinear.load(
                self.config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=False,
            )
            linear_names.append(f'{prefix}.q_proj')
        else:
            self.q_a_proj = TensorReplicatedLinear.load(
                self.config,
                prefix=f"{prefix}.q_a_proj",
                weights=weights,
                bias=self.config.attention_bias,
            )
            linear_names.append(f'{prefix}.q_a_proj')
            self.q_b_proj = TensorParallelColumnLinear.load(
                self.config,
                prefix=f"{prefix}.q_b_proj",
                weights=weights,
                bias=False,
            )
            linear_names.append(f'{prefix}.q_b_proj')

        self.kv_a_proj_with_mqa = TensorReplicatedLinear.load(
            self.config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
            weights=weights,
            bias=self.config.attention_bias,
        )
        linear_names.append(f'{prefix}.kv_a_proj_with_mqa')
        self.kv_a_layernorm = DeepseekV2RMSNorm(prefix=f"{prefix}.kv_a_layernorm", weights=weights)
        self.kv_b_proj = TensorParallelColumnLinear.load(
            self.config,
            prefix=f"{prefix}.kv_b_proj",
            weights=weights,
            bias=False,
        )
        linear_names.append(f'{prefix}.kv_b_proj')

        self.o_proj = TensorParallelRowLinear.load(
            self.config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=self.config.attention_bias,
        )
        linear_names.append(f'{prefix}.o_proj')

        self._init_rope(weights.device)

        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        weights.quantize = self.config.quantize
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        weights.quantize = config.quantize

        if self.q_lora_rank is not None:
            if self.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                            PackType.MIX_W8A8SC]:
                self.q_a_layernorm = DeepseekV2RMSNormBias(prefix=f"{prefix}.q_a_layernorm", weights=weights)
            else:
                self.q_a_layernorm = DeepseekV2RMSNorm(prefix=f"{prefix}.q_a_layernorm", weights=weights)

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling_dict is not None:
            mscale_all_dim = self.config.rope_scaling_dict.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling_dict["factor"]
            if mscale_all_dim:
                mscale = DeepseekV2YarnRotaryEmbedding.yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self, device):
        if self.config.rope_scaling_dict is None:
            self.rotary_emb = PositionRotaryEmbedding.static(dim=self.qk_rope_head_dim,
                                                             base=self.rope_theta,
                                                             device="cpu",
                                                             scaling_factor=self.scaling_factor).to(device)
        else:
            scaling_type = self.config.rope_scaling_dict["type"]
            scaling_factor = self.config.rope_scaling_dict["factor"]
            if scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling_dict[key]
                    for key in _ROPE_SCALING_KEYS
                    if key in self.config.rope_scaling_dict
                }
                yarn_kwargs = DeepseekV2YarnRotaryEmbedding.StaticInputArgs(
                                                            max_position_embeddings=self.max_position_embeddings,
                                                            scaling_factor=scaling_factor,
                                                            **kwargs,)
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding.static_yarn(dim=self.qk_rope_head_dim,
                                                                       base=self.rope_theta,
                                                                       device="cpu",
                                                                       yarn_kwargs=yarn_kwargs).to(device)
            else:
                msg = f"Unknown RoPE scaling type {scaling_type}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)


class FlashDeepseekV2DecoderLayer(nn.Module):
    def __init__(self, layer_idx, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.hidden_size = config.hidden_size

        self.self_attn = FlashDeepseekV2Attention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )

        self.mlp = (
            DeepseekV2MoE(prefix=f"{prefix}.mlp", config=config, weights=weights, shared_mlp_cls=DeepseekV2MLP)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        )
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.MIX_W8A16,
                                        PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                                        PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.input_layernorm = DeepseekV2RMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.input_layernorm = DeepseekV2RMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.input_layernorm = DeepseekV2RMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            self.input_layernorm = DeepseekV2RMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            msg = f'self_attn.pack_type: {self.self_attn.pack_type} not supported'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(msg)

        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.MIX_W8A16,
                                  PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                                  PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.post_attention_layernorm = DeepseekV2RMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.post_attention_layernorm = DeepseekV2RMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.post_attention_layernorm = DeepseekV2RMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            self.post_attention_layernorm = DeepseekV2RMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            msg = f'mlp.pack_type: {self.mlp.pack_type} not supported'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(msg)


class FlashDeepseekV2Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.parallel_embedding = False
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.v_head_dim = config.v_head_dim
        self.num_heads = config.num_attention_heads

        self.embed_tokens = (TensorParallelEmbedding if self.parallel_embedding else TensorEmbedding)(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashDeepseekV2DecoderLayer(
                    layer_idx,
                    config,
                    weights,
                    )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV2RMSNorm(prefix="model.norm", weights=weights, eps=config.rms_norm_eps)