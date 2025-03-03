# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# This file was copied from project `Qwen/Qwen-7B`

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorParallelEmbedding,
    load_column_multi,
    paged_attn,
    flash_attn,
    reshape_and_cache
)

from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.models.qwen2.modeling_base import QwenRMSNorm, QwenRMSNormBias, QwenRMSNormWrapper


class QwenMLP(nn.Module):
    def __init__(
            self, 
            prefix,
            config,
            weights
    ):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        linear_names = [f'{prefix}.w1', f'{prefix}.w2']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_2'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, None)
        self.w2_w1 = load_column_multi(
            config,
            prefixes=[f"{prefix}.w2", f"{prefix}.w1"],  # gate_up_proj
            weights=weights,
            head_size=1,
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",  # down_proj
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.w2_w1(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.c_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashQwenAttention(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        # can support self.num_heads % weights.process_group.size() != 0

        linear_names = [f'{prefix}.c_attn']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_1'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, None)
        self.c_attn = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.c_attn",
            weights=weights,
            bias=True,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=False,
        )

        self.prefix = prefix

    def forward(
            self,
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        qkv_combin = self.c_attn(hidden_states)
        query_combin, kv = qkv_combin.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_heads,
            ],
            dim=1,
        )
        query_combin = query_combin.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_heads, self.head_size)

        self.rotary_emb(query_combin, torch.select(kv, dim=1, index=0), cos, sin)

        reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query_combin)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn(
                query_combin,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attn(
                attn_output,
                query_combin,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.c_proj(attn_output.view(-1, self.num_heads * self.head_size))


class FlashQwenLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        self.attn = FlashQwenAttention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        if self.attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_1 = QwenRMSNorm(
                prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.ln_1 = QwenRMSNormBias(
                prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.ln_1 = QwenRMSNormWrapper(
                prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
            )
        else:
            msg = f'self_attn.pack_type: {self.self_attn.pack_type} not supported'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(msg)
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_2 = QwenRMSNorm(
                prefix=f"{prefix}.ln_2",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.ln_2 = QwenRMSNormBias(
                prefix=f"{prefix}.ln_2",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.ln_2 = QwenRMSNormWrapper(
                prefix=f"{prefix}.ln_2", weights=weights, eps=config.layer_norm_epsilon
            )
        else:
            msg = f'mlp.pack_type: {self.mlp.pack_type} not supported'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(msg)

    def forward(
            self,
            hidden_states,
            residual,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        normed_hidden_states_combin, res = self.ln_1(hidden_states, residual)

        # Self Attention
        attn_output = self.attn(
            normed_hidden_states_combin,
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
        normed_attn_res_output_combin, attn_res = self.ln_2(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output_combin)

        return mlp_output, attn_res


class FlashQwenModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorParallelEmbedding(
            prefix="transformer.wte", weights=weights
        )
        self.h = nn.ModuleList(
            [
                FlashQwenLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = QwenRMSNorm(
            prefix="transformer.ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads
