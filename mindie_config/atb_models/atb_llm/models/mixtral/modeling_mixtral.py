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
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from atb_llm.models.base.modeling import FlashAttention
from atb_llm.utils.layers.linear import FastLinear, TensorReplicatedLinear
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    load_column_multi,
    RMSNorm
    )
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.gate_up_proj = load_column_multi(
            config,
            prefixes=[f"{prefix}.w1", f"{prefix}.w3"],
            weights=weights,
            head_size=1,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.w2",
            weights=weights,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]


class MixtralSparseMoeBlock(nn.Module):
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
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = True  # defaulting the model to tensor parallel
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        if (self.expert_parallel_degree == 0):
            msg = "expert parallel degree should not be 0!"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        self.expert_lists = []
        if self.tp:
            self.expert_lists = [[i for i in range(config.n_routed_experts)] for j in range(self.tp_world_size)]
        else:
            self.expert_lists = assign(config.n_routed_experts, self.tp_world_size)
        expert_prefix = f"{prefix}.experts"
        self.gate = FastLinear.load(
                prefix=f"{prefix}.gate",
                weights=weights,
                bias=False,
                )
        linear_names = [f'{expert_prefix}.{0}.w1', f'{expert_prefix}.{0}.w3']
        pack_name = f'{expert_prefix}.{0}.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.tp:
            if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                self.gate_up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_up_proj.append(load_column_multi(
                        config,
                        prefixes=[f"{expert_prefix}.{i}.w1", f"{expert_prefix}.{i}.w3"],
                        weights=weights,
                        head_size=1,
                    ))
            elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
                self.gate_up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_up_proj.append(TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{expert_prefix}.{i}.gate_up_proj",
                    weights=weights,
                    bias=False,
                    ))
            else:
                self.gate_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_proj.append(TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{expert_prefix}.{i}.w1",
                    weights=weights,
                    bias=False,
                    ))
                self.up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.up_proj.append(TensorParallelColumnLinear.load(
                        config,
                        prefix=f"{expert_prefix}.{i}.w3",
                        weights=weights,
                        bias=False,
                    ))
                
            self.down_proj = nn.ModuleList()
            for i in range(self.num_experts):
                self.down_proj.append(TensorParallelRowLinear.load(
                config,
                prefix=f"{expert_prefix}.{i}.w2",
                weights=weights,
                bias=False,
                ))
            self.intermediate_size = (
                    (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
            )
   
        else:
            if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                self.gate_up_proj = nn.ModuleList()
                for i in self.expert_lists[self.tp_rank]:
                    self.gate_up_proj.append(TensorReplicatedLinear.load(
                        config,
                        prefixes=[f"{expert_prefix}.{i}.w1", f"{expert_prefix}.{i}.w3"],
                        weights=weights,
                        head_size=1,
                    ))
            self.down_proj = nn.ModuleList()
            for i in self.expert_lists[self.tp_rank]:
                self.down_proj.append(TensorReplicatedLinear.load(
                config,
                prefix=f"{expert_prefix}.{i}.w2",
                weights=weights,
                bias=False,
                ))

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        return final_hidden_states, router_logits


class FlashMixtralAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)

        super().load_qkv_weights(**kwargs)


class FlashMixtralLayer(nn.Module):
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
        self.self_attn = FlashMixtralAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.block_sparse_moe = MixtralSparseMoeBlock(
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


class FlashMixtralModel(torch.nn.Module):
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
                FlashMixtralLayer(
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

 