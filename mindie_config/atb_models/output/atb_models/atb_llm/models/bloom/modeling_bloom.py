# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

import math

import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear
)

from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from .config_bloom import BloomConfig


class TensorParallelColumnEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=1)

        self.process_group = weights.process_group
        self.reduce = reduce

        self.weight = nn.Parameter(weight)


class BloomLayerNorm(nn.Module):
    def __init__(self, prefix, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)


class BloomAttention(nn.Module):
    def __init__(self, prefix, config: BloomConfig, weights):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.process_group = weights.process_group
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            msg = f"Hidden size must be divisible by number of heads (got hidden size: {self.hidden_size}" + \
                  f" and number of heads: {self.num_heads})."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        process_group = weights.process_group
        if self.num_heads % process_group.size() != 0:
            msg = f"Number of heads must be divisible by number of shards (got number of heads: {self.num_heads}" + \
                  f" and number of shards: {process_group.size()})."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        self.num_heads = self.num_heads // process_group.size()
        self.query_key_value = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=True,
        )

        self.dense = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.dense", weights=weights, bias=True, bias_pre_add=True
        )
        self.o_proj = self.dense
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        linear_names = [f'{prefix}.query_key_value']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        else:
            self.pack_type = PackType.ALL_FP


class BloomMLP(nn.Module):
    def __init__(self, prefix, config: BloomConfig, weights):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True, bias_pre_add=True
        )
        self.gelu_impl = torch.nn.GELU(approximate="tanh")
        self.hidden_dropout = config.hidden_dropout

        linear_names = [f'{prefix}.dense_h_to_4h', f'{prefix}.dense_4h_to_h']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'

        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        else:
            self.pack_type = PackType.ALL_FP


class BloomBlock(nn.Module):
    def __init__(self, layer_id: int, config: BloomConfig, weights, prefix):
        super().__init__()

        self.input_layernorm = BloomLayerNorm(
            prefix=f"{prefix}h.{layer_id}.input_layernorm",
            weights=weights
        )
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(
            prefix=f"{prefix}h.{layer_id}.self_attention", config=config, weights=weights
        )
        self.post_attention_layernorm = BloomLayerNorm(
            prefix=f"{prefix}h.{layer_id}.post_attention_layernorm",
            weights=weights
        )

        self.mlp = BloomMLP(prefix=f"{prefix}h.{layer_id}.mlp", config=config, weights=weights)
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout


class FlashBloomModel(nn.Module):
    def __init__(self, config: BloomConfig, weights):
        super().__init__()
        extra_prefix = "" if not config.quantize else "transformer."  # msmodelslim 量化后的模型会多一个 transformer 的前缀
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.word_embeddings = TensorParallelColumnEmbedding(
            prefix=f"{extra_prefix}word_embeddings", weights=weights
        )

        self.word_embeddings_layernorm = BloomLayerNorm(
            prefix=f"{extra_prefix}word_embeddings_layernorm",
            weights=weights
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                BloomBlock(layer_id=layer_id, config=config, weights=weights, prefix=f"{extra_prefix}")
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        # Final Layer Norm
        self.ln_f = BloomLayerNorm(
            prefix=f"{extra_prefix}ln_f", weights=weights
        )

        self.head_size = config.hidden_size // config.n_head
        self.num_heads = config.n_head
        self.num_key_value_heads = config.n_head
