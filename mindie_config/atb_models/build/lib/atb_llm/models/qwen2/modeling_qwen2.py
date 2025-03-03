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
    FA3,
    paged_attn,
    flash_attn,
    reshape_and_cache
)

from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.models.qwen2.modeling_base import (
    QwenRMSNorm,
    QwenRMSNormBias,
    QwenRMSNormWrapper,
    QwenRMSNormAntiOutlierWrapper
)


class QwenMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        pack_name = f'{prefix}.w2_w1'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI]:
            self.w2_w1 = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.w2_w1 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.w2_w1",
                weights=weights,
                bias=False,
            )
        else:
            self.w2 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_proj",
                weights=weights,
                bias=False,
            )
            self.w1 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.up_proj",
                weights=weights,
                bias=False,
            )
        if config.quantize == "w8a8sc":
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",  # down_proj
                weights=weights,
                bias=False,
            )
        else:
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.down_proj",  # down_proj
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
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_world_size = process_group.size()
        self.num_kv_heads = config.num_key_value_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        if config.quantization_config.kv_quant_type is not None:
            self.kv_cache_quant = KvCache.load(prefix_k=f"{prefix}.k_proj", prefix_v=f"{prefix}.v_proj",
                                               weights=weights)

        if config.quantization_config.fa_quant_type is not None:
            self.fa3 = FA3.load(prefix_q=f"{prefix}.fa_q", prefix_k=f"{prefix}.fa_k", prefix_v=f"{prefix}.fa_v",
                weights=weights, head_size=self.head_size)

        self.softmax_scale = self.head_size ** -0.5

        # can support self.num_heads % weights.process_group.size() != 0
        linear_names = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        pack_name = f'{prefix}.c_attn'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if (self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI]
            and config.num_attention_heads % self.tp_world_size == 0):
            self.c_attn = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size,
                bias=True
            )
            self.c_attn.linear.num_linear_before_pack = 3
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.c_attn = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.c_attn",
                weights=weights,
                bias=True,
            )
            self.c_attn.linear.num_linear_before_pack = 3
        else:
            if config.num_attention_heads % self.tp_world_size != 0:
                self.c_attn = TensorParallelColumnLinear.load_column_multi_c(
                    config,
                    prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                    weights=weights,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads
                )
            else:
                self.q_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.q_proj",
                    weights=weights,
                    bias=True,
                )
                self.k_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.k_proj",
                    weights=weights,
                    bias=True,
                )
                self.v_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.v_proj",
                    weights=weights,
                    bias=True,
                )
        if config.quantize == "w8a8sc":
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=False,
            )
        else:
            if config.num_attention_heads % self.tp_world_size != 0:
                self.c_proj = TensorParallelColumnLinear.load_o(
                    config,
                    prefix=f"{prefix}.o_proj",
                    weights=weights,
                    bias=False,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads
                )
            else:
                self.c_proj = TensorParallelRowLinear.load(
                    config,
                    prefix=f"{prefix}.o_proj",
                    weights=weights,
                    bias=False,
                    gqa_size=self.head_size
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
        qkv = self.c_attn(hidden_states)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn(
                query,
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
                query,
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
    def __init__(self, layer_id, config, weights, prefix):
        super().__init__()
        quantize_type = "w8a8sc"  # w8a8sc：稀疏量化
        if config.quantize == quantize_type:
            prefix = f"transformer.h.{layer_id}"
        else:
            prefix = f"{prefix}.layers.{layer_id}"
            
        if config.quantize == quantize_type:
            self.attn = FlashQwenAttention(
                prefix=f"{prefix}.attn", config=config, weights=weights
            )
            self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        else:
            self.attn = FlashQwenAttention(
                prefix=f"{prefix}.self_attn", config=config, weights=weights
            )
            self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        
        if self.attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16, PackType.ALL_W4A16]:
            self.ln_1 = QwenRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI]:
            self.ln_1 = QwenRMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            if config.quantize == quantize_type:
                self.ln_1 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.ln_1", weights=weights, eps=config.rms_norm_eps
                )
            else:
                self.ln_1 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
                )
        elif self.attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            if config.quantize == quantize_type:
                self.ln_1 = QwenRMSNormBias(
                    prefix=f"{prefix}.ln_1", weights=weights, eps=config.rms_norm_eps
                )
            else:
                self.ln_1 = QwenRMSNormBias(
                    prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
                )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.ln_2 = QwenRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI]:
            self.ln_2 = QwenRMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            if config.quantize == quantize_type:
                self.ln_2 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.ln_2", weights=weights, eps=config.rms_norm_eps
                )
            else:
                self.ln_2 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
                )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            if config.quantize == quantize_type:
                self.ln_2 = QwenRMSNormBias(
                    prefix=f"{prefix}.ln_2",
                    weights=weights,
                    eps=config.rms_norm_eps,
                )
            else:
                self.ln_2 = QwenRMSNormBias(
                    prefix=f"{prefix}.post_attention_layernorm",
                    weights=weights,
                    eps=config.rms_norm_eps,
                )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')

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
        normed_hidden_states, res = self.ln_1(hidden_states, residual)

        # Self Attention
        attn_output = self.attn(
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
        normed_attn_res_output, attn_res = self.ln_2(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashQwenModel(torch.nn.Module):
    def __init__(self, config, weights, **kwargs):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        model_prefix = kwargs.get("model_prefix", "model")
        if config.quantize == "w8a8sc":
            self.wte = TensorEmbedding(
                prefix="transformer.wte", weights=weights
            )
        else:
            self.wte = TensorParallelEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )
        self.h = nn.ModuleList(
            [
                FlashQwenLayer(
                    layer_id,
                    config,
                    weights,
                    model_prefix,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        if config.quantize == "w8a8sc":
            self.ln_f = QwenRMSNorm(
                prefix="transformer.ln_f", weights=weights, eps=config.rms_norm_eps
            )
        else:
            self.ln_f = QwenRMSNorm(
                prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps
            )
        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads
