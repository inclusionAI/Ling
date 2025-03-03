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

from typing import Optional, List, Tuple

import torch
import torch.distributed
from torch import nn
from atb_llm.utils.layers import TensorEmbedding, RMSNorm
from atb_llm.models.base.modeling import FlashAttention
from atb_llm.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


class SkyworkSparseMoeBlock(MixtralSparseMoeBlock):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)


class FlashSkyworkAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)

        super().load_qkv_weights(**kwargs)


class FlashSkyworkLayer(nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                     hidden_states: torch.tensor,
                     residual: torch.tensor,
                     cos: torch.tensor,
                     sin: torch.tensor,
                     cu_seqlen_prefill: torch.tensor,
                     kv_cache: Tuple[torch.tensor, torch.tensor],
                     block_tables: List[torch.tensor],
                     slots: torch.tensor,
                     input_lengths: torch.tensor,
                     max_s: torch.tensor):
            self.hidden_states = hidden_states
            self.residual = residual
            self.cos = cos
            self.sin = sin
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s

    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashSkyworkAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.block_sparse_moe = SkyworkSparseMoeBlock(
            prefix=f"{prefix}.block_sparse_moe", config=config, weights=weights
        )
        self.input_layernorm = RMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
            self,
            input_args: ForwardInputArgs
    ):
        hidden_states = input_args.hidden_states
        residual = input_args.residual
        cos = input_args.cos
        sin = input_args.sin
        cu_seqlen_prefill = input_args.cu_seqlen_prefill
        kv_cache = input_args.kv_cache
        block_tables = input_args.block_tables
        slots = input_args.slots
        input_lengths = input_args.input_lengths
        max_s = input_args.max_s
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashSkyworkModel(torch.nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                    input_ids: torch.Tensor,
                    position_ids: torch.Tensor,
                    cu_seqlen_prefill: Optional[torch.Tensor],
                    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                    block_tables: torch.Tensor,
                    slots: torch.Tensor,
                    input_lengths: torch.Tensor,
                    max_s: int,
                    lm_head_indices: Optional[torch.Tensor] = None):
            self.input_ids = input_ids
            self.position_ids = position_ids
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s
            self.lm_head_indices = lm_head_indices

    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashSkyworkLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

 