# coding=utf-8
#  Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelEmbedding,
    load_column_multi,
    KvCache,
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.config import BaseConfig


class MiniCpmConfig(BaseConfig):
    model_type = "minicpm"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pretraining_tp=1,
            tie_word_embeddings=False,
            skip_word_embedding=False,
            pe_type="ROPE",
            rope_vanilla_theta=None,
            rope_mscale=None,
            rope_keep_local_base_windows=None,
            rope_given_inv_feq_str=None,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            scale_emb=1,
            dim_model_base=1,
            scale_depth=1,
            alibi_bias_max=8.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.skip_word_embedding = skip_word_embedding
        self.pe_type = pe_type
        self.rope_vanilla_theta = rope_vanilla_theta
        self.rope_mscale = rope_mscale
        self.rope_keep_local_base_windows = rope_keep_local_base_windows
        self.rope_given_inv_feq_str = rope_given_inv_feq_str
        self.alibi_bias_max = alibi_bias_max
        self.attention_bias = attention_bias
        self.scale_depth = scale_depth
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings,
                        **kwargs,)

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            msg = f"Rope scaling must be a dictionary with with two fields, `type` and `factor`," + \
                  f" got {self.rope_scaling}."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            msg = f"Type field of rope scaling must be one of ['linear', 'dynamic']," + \
                  f" got {rope_scaling_type}."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            msg = f"Factor of rope scaling must be a float > 1," + \
                  f" got {rope_scaling_factor}."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)


class MiniCpmRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        MiniCpmRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class MiniCpmMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        pack_name = f'{prefix}.gate_up_proj'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI
        ]:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )

        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                bias=False,
                weights=weights,
            )

        else:
            self.gate_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_proj",
                weights=weights,
                bias=False,
            )
            self.up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.up_proj",
                bias=False,
                weights=weights,
            )

        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class FlashMiniCpmAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        if config.quantization_config.kv_quant_type is not None:
            self.k_cache_quant = KvCache.load(prefix=f"{prefix}.k_proj", weights=weights)
            self.v_cache_quant = KvCache.load(prefix=f"{prefix}.v_proj", weights=weights)
        
        linear_names = [f'{prefix}.q_proj', f'{prefix}.k_proj', f'{prefix}.v_proj']
        pack_name = f'{prefix}.query_key_value'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        self.softmax_scale = self.head_size ** -0.5

        # can not support self.num_heads % weights.process_group.size() != 0
        if (config.num_attention_heads != config.num_key_value_heads
                and (self.num_heads % weights.process_group.size() != 0)):
            msg = f"Number of heads must be divisible by number of shards (got number of heads: {self.num_heads}" + \
                  f" and number of shards: {weights.process_group.size()})."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        if config.num_key_value_heads < weights.process_group.size():
            repeat_times = weights.process_group.size() // config.num_key_value_heads
        else:
            repeat_times = 1

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads * repeat_times
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads

        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI
        ]:
            self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj",
                          f"{prefix}.k_proj",
                          f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size
            )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.query_key_value = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.query_key_value",
                weights=weights,
                bias=False,
            )
        else:
            self.q_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=False,
            )
            self.k_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.k_proj",
                weights=weights,
                bias=False,
            )
            self.v_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.v_proj",
                weights=weights,
                bias=False,
            )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix


class FlashMiniCpmLayer(nn.Module):
    def __init__(self, layer_id, config, weights, model_prefix="model"):
        super().__init__()
        prefix = f"{model_prefix}.layers.{layer_id}"
        self.self_attn = FlashMiniCpmAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = MiniCpmMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.input_layernorm = MiniCpmRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            msg = f"Not support pack type of self attention: {self.self_attn.pack_type}"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.post_attention_layernorm = MiniCpmRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            msg = f"Not support pack type of mlp:  {self.mlp.pack_type}"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)


class FlashMiniCpmModel(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="model"):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.parallel_embedding = False
        self.embed_tokens = (TensorParallelEmbedding if self.parallel_embedding else TensorEmbedding)(
            prefix=f"{model_prefix}.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashMiniCpmLayer(
                    layer_id,
                    config,
                    weights,
                    model_prefix
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniCpmRMSNorm(prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads


class MiniCpmModel(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="model"):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix=f"{model_prefix}.embed_tokens", weights=weights
        )
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [
                FlashMiniCpmLayer(
                    layer_id,
                    config,
                    weights,
                    model_prefix
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniCpmRMSNorm(
            prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads
