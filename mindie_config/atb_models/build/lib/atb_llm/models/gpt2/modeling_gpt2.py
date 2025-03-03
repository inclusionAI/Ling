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
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    RMSNormBias,
)
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type
from torch import nn
from atb_llm.models.base.modeling import FlashLayer


class GPT2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        linear_names = [f"{prefix}.c_fc"]
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.ln_2'
        self.pack_type = calc_linear_pack_type(weights, linear_names, self.norm_name)
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.c_fc",
            weights=weights,
            bias=True,
            dim=1
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=True,
            bias_pre_add=True,
            dim=0
        )


class FlashGPT2Attention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()

        linear_names = [f'{prefix}.c_attn']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.ln_1'
        self.pack_type = calc_linear_pack_type(weights, linear_names, self.norm_name)
        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.c_attn",
            weights=weights,
            bias=True,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,
            dim=1
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=True,
            bias_pre_add=True,
            dim=0
        )


class FlashGPT2Layer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.prefix = f"h.{layer_id}"
        self.norm_bias = True
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashGPT2Attention(
            prefix=f"{self.prefix}.attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = GPT2MLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashGPT2Model(torch.nn.Module):
    def __init__(self, config, weights, **kwargs):
        super().__init__()
        self.wte = TensorEmbedding(prefix="wte", weights=weights)
        self.wpe = TensorEmbedding(prefix="wpe", weights=weights)
        self.layers = nn.ModuleList(
            [
                FlashGPT2Layer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNormBias(
            prefix="ln_f", weights=weights, eps=config.layer_norm_epsilon
        )
