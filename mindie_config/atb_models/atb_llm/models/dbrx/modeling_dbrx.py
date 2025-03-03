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

from atb_llm.utils.layers.linear import FastLinear

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelEmbedding
)

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type


class DbrxLayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        DbrxLayerNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class DbrxLayerNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        DbrxLayerNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class DbrxLayerNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        DbrxLayerNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.ori = DbrxLayerNorm(prefix, weights, eps)
        self.anti = DbrxLayerNormBias(f'{prefix}.module', weights, eps)


class DbrxLayerNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        DbrxLayerNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.ori = DbrxLayerNorm(f'{prefix}.ori', weights, eps)
        self.anti = DbrxLayerNormBias(f'{prefix}.anti', weights, eps)


class FlashDbrxAttention(nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()

        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.resid_pdrop
        self.head_size = self.hidden_size // self.num_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            msg = f"Hidden size must be divisible by number of heads (got hidden size: {self.hidden_size}" + \
                  f" and number of heads: {self.num_heads})."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        self.query_key_value = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.Wqkv",
            weights=weights,
            bias=False,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads
        )

        self.o_proj = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.out_proj", weights=weights, bias=False, bias_pre_add=False
        )
        self.attention_dropout = nn.Dropout(config.resid_pdrop)

        linear_names = [f'{prefix}.Wqkv']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.norm_1'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        else:
            self.pack_type = PackType.ALL_FP


class DbrxExpertGate(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)


class GetOneWeight(FastLinear):
    def __init__(self, prefix: str, weights, index, reduce=True):
        weight = weights.get_partial_sharded_mid_dim(f"{prefix}", dim=0, index=index)
        super().__init__(weight, bias=None)
        self.process_group = weights.process_group
        self.reduce = reduce
        self.weight = nn.Parameter(weight)


class GetTwoWeight(FastLinear):
    def __init__(self, prefix1: str, prefix2: str, weights, index, reduce=True):
        weight1 = weights.get_partial_sharded_mid_dim(f"{prefix1}", dim=0, index=index)
        weight2 = weights.get_partial_sharded_mid_dim(f"{prefix2}", dim=0, index=index)
        all_weight = torch.cat([weight1, weight2], dim=0)
        super().__init__(all_weight, bias=None)
        self.process_group = weights.process_group
        self.reduce = reduce

        self.weight = nn.Parameter(all_weight)


class DbrxMoe(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = False
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        if self.expert_parallel_degree == 0:
            msg = f"Expert parallel degree should not be 0!"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        self.num_experts = config.ffn_config.moe_num_experts

        self.config = config
        self.num_experts_per_tok = config.ffn_config.moe_top_k
        expert_prefix = f"{prefix}.experts.mlp"
        self.experts = nn.ModuleList()
        gate_prefix = f"{prefix}.router.layer"

        self.gate = FastLinear.load(
            prefix=f"{gate_prefix}",
            weights=weights,
            bias=False,
        )

        linear_names = [f'{expert_prefix}.w1', f'{expert_prefix}.v1']
        pack_name = f'{expert_prefix}.w1_v1'
        layer_prefix = '.'.join(expert_prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.norm_2'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI
        ]:
            self.w1_v1 = nn.ModuleList()
            for i in range(self.num_experts):
                self.w1_v1.append(
                    GetTwoWeight(prefix1=f"{expert_prefix}.w1", prefix2=f"{expert_prefix}.v1",
                                 weights=weights, index=i))

            self.w2 = nn.ModuleList()
            for i in range(self.num_experts):
                self.w2.append(GetOneWeight(prefix=f"{expert_prefix}.w2", weights=weights, index=i))

            intermediate_size = config.ffn_config.ffn_hidden_size
            self.intermediate_size = (
                    (intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
            )


class FlashDbrxLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        if config.quantize == "w8a8sc":
            prefix = f"transformer.blocks.{layer_id}"
        else:
            prefix = f"transformer.blocks.{layer_id}"

        if config.quantize == "w8a8sc":
            self.self_attn = FlashDbrxAttention(
                prefix=f"{prefix}.norm_attn_norm.attn", config=config, weights=weights
            )
            self.mlp = DbrxMoe(prefix=f"{prefix}.mlp", config=config, weights=weights)
        else:
            self.self_attn = FlashDbrxAttention(
                prefix=f"{prefix}.norm_attn_norm.attn", config=config, weights=weights
            )
            self.mlp = DbrxMoe(prefix=f"{prefix}.ffn", config=config, weights=weights)

        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_1 = DbrxLayerNormBias(
                prefix=f"{prefix}.norm_attn_norm.norm_1", weights=weights
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.ln_1 = DbrxLayerNormWrapper(
                prefix=f"{prefix}.norm_attn_norm.norm_1", weights=weights
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            if config.quantize == "w8a8sc":
                self.ln_1 = DbrxLayerNormAntiOutlierWrapper(
                    prefix=f"{prefix}.norm_attn_norm.norm_1", weights=weights
                )
            else:
                self.ln_1 = DbrxLayerNormAntiOutlierWrapper(
                    prefix=f"{prefix}.norm_attn_norm.norm_1", weights=weights
                )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            if config.quantize == "w8a8sc":
                self.ln_1 = DbrxLayerNormBias(
                    prefix=f"{prefix}.norm_attn_norm.norm_1", weights=weights
                )
            else:
                self.ln_1 = DbrxLayerNormBias(
                    prefix=f"{prefix}.norm_attn_norm.norm_1", weights=weights
                )
        else:
            msg = f"Pack type of self attention: {self.self_attn.pack_type} not supported"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        self.ln_2 = DbrxLayerNormBias(
            prefix=f"{prefix}.norm_attn_norm.norm_2",
            weights=weights,
        )


class FlashDbrxModel(nn.Module):
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
        if config.quantize == "w8a8sc":
            self.wte = TensorEmbedding(
                prefix="transformer.wte", weights=weights
            )
        else:

            self.wte = TensorParallelEmbedding(
                prefix="transformer.wte", weights=weights
            )

        self.layers = nn.ModuleList(
            [
                FlashDbrxLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads

        if config.quantize == "w8a8sc":
            self.ln_f = DbrxLayerNormBias(
                prefix="transformer.norm_f", weights=weights
            )
        else:
            self.ln_f = DbrxLayerNormBias(
                prefix="transformer.norm_f", weights=weights
            )
