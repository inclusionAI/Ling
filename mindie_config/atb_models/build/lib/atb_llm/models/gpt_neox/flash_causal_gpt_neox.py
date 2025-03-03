# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
import json
import math
import os
from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch_npu
from atb_llm.utils.log.error_code import ErrorCode
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from atb_llm.models.gpt_neox.config_gpt_neox import GPTNeoXConfig

from atb_llm.utils.env import ENV
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    TensorParallelHead,
    PositionRotaryEmbedding,
    AttentionMask,
    get_linear,
)
from atb_llm.utils.log import logger


def load_ascend_transformer():
    atb_speed_home_path = ENV.atb_speed_home_path
    if atb_speed_home_path is None:
        error_msg = "env ATB_SPEED_HOME_PATH not exist, source set_env.sh"
        logger.error(error_msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
        raise ValueError(error_msg)
    lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
    logger.info(f"load {lib_path}")
    torch.classes.load_library(lib_path)


# for ascend init
load_ascend_transformer()


def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

    if bias:
        bias = weights.get_tensor(f"{prefix}.bias")

    bias = bias / weights.process_group.size()

    linear = get_linear(weight, bias, config.quantize)
    # different all_reduce positions
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelRowLinear(linear, process_group=weights.process_group)


def load_qkv(config, prefix: str, weights, num_heads, head_size, hidden_size):
    weight = weights.get_multi_weights_col([prefix], quantize=config.quantize, dim=0)
    if isinstance(weight, torch.Tensor):
        # Only on non quantized versions
        weight = (
            weight.view(
                num_heads,
                3,
                head_size,
                hidden_size,
            )
            .permute(1, 0, 2, 3)
            .reshape(-1, hidden_size)
        )

    bias = weights.get_sharded(f"{prefix}.bias", dim=0)
    bias = bias.view(num_heads, 3, head_size).permute(1, 0, 2).reshape(-1)

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelColumnLinear(linear)


class FastLayerNorm(torch.nn.Module):
    def __init__(self, prefix, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states

        return super(FastLayerNorm, self).forward(hidden_states), residual


class FlashNeoxAttention(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        if self.num_heads % weights.process_group.size() != 0:
            error_msg = f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} " \
                        f"and `num_shards`: {weights.process_group.size()}"
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)
        self.num_heads = self.num_heads // weights.process_group.size()

        self.rotary_dims = int(self.head_size * config.rotary_pct)
        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.rotary_dims, base=config.rotary_emb_base,
                                                         device="cpu").to(weights.device)

        self.softmax_scale = self.head_size ** (-0.5)

        self.query_key_value = load_qkv(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            num_heads=self.num_heads,
            head_size=self.head_size,
            hidden_size=self.hidden_size,
        )
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=True
        )
        self.kv_head_mapping = torch.arange(
            0, self.num_heads, dtype=torch.int32, device=weights.device
        )
        self.num_key_value_heads = self.num_heads
        self.prefix = prefix


class FlashMLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        act = config.hidden_act
        self.act = ACT2FN[act]

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = load_row(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True
        )


class FlashNeoXLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        prefix = f"gpt_neox.layers.{layer_id}"

        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = FastLayerNorm(
            prefix=f"{prefix}.input_layernorm", 
            weights=weights
        )
        self.post_attention_layernorm = FastLayerNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights
        )
        self.attention = FlashNeoxAttention(
            config, prefix=f"{prefix}.attention", weights=weights
        )

        self.mlp = FlashMLP(config, prefix=f"{prefix}.mlp", weights=weights)
        self.process_group = weights.process_group


class FlashGPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPTNeoXModel(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config)
        self.config = config

        self.embed_in = TensorEmbedding(
            prefix="gpt_neox.embed_in", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = FastLayerNorm(
            prefix="gpt_neox.final_layer_norm",
            weights=weights
        )

        self.gradient_checkpointing = False
        self.soc_info = NPUSocInfo()

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)


class FlashGpt_neoxForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config)
        self.gpt_neox = FlashGPTNeoXModel(config, weights)

        self.embed_out = TensorParallelHead.load(
            config, prefix="embed_out", weights=weights
        )

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        if self.num_heads % self.tp_world_size != 0:
            error_msg = f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} " \
                        f"and `num_shards`: {self.tp_world_size}"
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)
        self.num_heads = self.num_heads // self.tp_world_size
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = self.num_heads

        self.soc_info = NPUSocInfo()

        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.rotary_dims = int(self.head_size * config.rotary_pct)
        self.ascend_rotary_embedding = PositionRotaryEmbedding.static(dim=self.rotary_dims, base=config.rotary_emb_base,
                                                                      device="cpu").to(weights.device)
        self.init_ascend_operations(config)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

    def maybe_format_cast(self, tensor):
        """
        maybe_format_cast
        """
        if not self.soc_info.need_nz:  # transdata 会额外占资源
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def init_ascend_operations(self, config):
        self.acl_param_encoder = json.dumps({
            "headNum": self.num_heads,
            "dk": self.head_size,
            "layerNum": config.num_hidden_layers,
            "layerNormEps": config.layer_norm_eps,
            "rotaryPct": config.rotary_pct,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "qScale": 1 / math.sqrt(self.head_size),
            "qkScale": 1.0,
            "backend": self.soc_info.communication_backend,
        })
        self.acl_param_decoder = json.dumps({
            "headNum": self.num_heads,
            "dk": self.head_size,
            "layerNum": config.num_hidden_layers,
            "layerNormEps": config.layer_norm_eps,
            "rotaryPct": config.rotary_pct,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "qScale": 1 / math.sqrt(self.head_size),
            "qkScale": 1.0,
            "backend": self.soc_info.communication_backend,
        })

        self.max_position_embeddings = config.max_position_embeddings
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("gptneox_20b_PAModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("gptneox_20b_PAModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers

        self.acl_encoder_operation_inputs: list = [None] * 9
        self.acl_decoder_operation_inputs: list = [None] * 9
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64)

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)

    def init_ascend_weight(self):
        weights = [self.gpt_neox.state_dict()["embed_in.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.gpt_neox.layers[i].state_dict()
            weights_t.append(weights_layer['input_layernorm.weight'])
            weights_t.append(weights_layer['input_layernorm.bias'])
            weights_t.append(weights_layer['post_attention_layernorm.weight'])
            weights_t.append(weights_layer['post_attention_layernorm.bias'])
            weights_t.append(self.maybe_format_cast(weights_layer['attention.query_key_value.weight']))
            weights_t.append(weights_layer['attention.query_key_value.bias'])
            weights_t.append(self.maybe_format_cast(weights_layer['attention.dense.weight']))
            weights_t.append(weights_layer['attention.dense.bias'])
            weights_t.append(self.maybe_format_cast(weights_layer['mlp.dense_h_to_4h.linear.weight']))
            weights_t.append(weights_layer['mlp.dense_h_to_4h.linear.bias'])
            weights_t.append(self.maybe_format_cast(weights_layer['mlp.dense_4h_to_h.weight']))
            weights_t.append(weights_layer['mlp.dense_4h_to_h.bias'])
            weights.extend(weights_t)
            if self.soc_info.need_nz:
                del self.gpt_neox.layers[i].attention
                del self.gpt_neox.layers[i].mlp
                del self.gpt_neox.layers[i].post_attention_layernorm

        weights.append(self.gpt_neox.state_dict()["final_layer_norm.weight"])
        weights.append(self.gpt_neox.state_dict()["final_layer_norm.bias"])
        weights.append(self.maybe_format_cast(self.state_dict()["embed_out.linear.weight"]))

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

    def init_ascend_kvcache(self, kv_cache):
        flag = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0]) \
               or not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if flag:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            logger.debug(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.debug(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        cos_embed, sin_embed = self.ascend_rotary_embedding.get_cos_sin_total(
            position_ids, self.max_position_embeddings, torch.float16
        )

        if self.soc_info.need_nz:
            pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
            atten_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
            atten_mask = self.transdata_operation.execute([atten_mask])[0]
        else:
            atten_mask = self.ascend_atten_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                              kv_cache[0][0].device)
        if is_prefill:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param_encoder = json.dumps({
                "seqLen": input_lengths.tolist()
            })

            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids
            self.acl_encoder_operation_inputs[2] = cos_embed
            self.acl_encoder_operation_inputs[3] = sin_embed
            self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[8] = lm_head_indices.to(torch.int64)
            return self.acl_encoder_operation_inputs, self.acl_param_encoder
        else:
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids
            self.acl_decoder_operation_inputs[2] = cos_embed
            self.acl_decoder_operation_inputs[3] = sin_embed
            self.acl_decoder_operation_inputs[4] = atten_mask
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.lm_head_indices_fake.to(input_ids.device)
            self.acl_param_decoder = json.dumps({
                "seqLen": input_lengths.tolist()
            })

            return self.acl_decoder_operation_inputs, self.acl_param_decoder

    def execute_ascend_operator(self,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_seq_len: int,
                                lm_head_indices: Optional[torch.Tensor] = None,
                                **kwargs):
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)

        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:

        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)
        logits = self.execute_ascend_operator(input_ids, position_ids, is_prefill, kv_cache,
                                              block_tables, slots, input_lengths,
                                              max_seq_len, lm_head_indices, **kwargs)
        return logits

