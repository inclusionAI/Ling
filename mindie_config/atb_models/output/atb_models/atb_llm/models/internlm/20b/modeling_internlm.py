# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
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
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import torch
from torch import nn

from atb_llm.models.base.modeling import get_suffix, FlashAttention, MLP, FlashLayer
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorEmbedding,
    RMSNorm,
    load_column_multi,
)
from atb_llm.utils.quantize.pack_type import PackType


class InternlmMLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)

    def load_gate_up_weights(self, **kwargs):
        if self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W8A16]:
            gate_up_linear = load_column_multi(
                self.config,
                prefixes=self.gate_up_names,
                weights=self.weights,
                head_size=1,
            )
            setattr(self, get_suffix(self.pack_name), gate_up_linear)
        else:
            for name in self.gate_up_names:
                linear = TensorParallelColumnLinear.load(
                    self.config,
                    prefix=name,
                    weights=self.weights,
                    bias=False,
                )
                setattr(self, get_suffix(name), linear)


class FlashInternlmAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)

    def load_qkv_weights(self, **kwargs):
        if self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W8A16]:
            query_key_value_linear = load_column_multi(
                self.config,
                prefixes=self.qkv_names,
                weights=self.weights,
                head_size=self.head_size
            )
            setattr(self, get_suffix(self.pack_name), query_key_value_linear)
        else:
            for name in self.qkv_names:
                linear = TensorParallelColumnLinear.load(
                    self.config,
                    prefix=name,
                    weights=self.weights,
                    bias=False,
                )
                setattr(self, get_suffix(name), linear)


class FlashInternlmLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="model", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashInternlmAttention(
            prefix=f"{self.prefix}.self_attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = InternlmMLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashInternlmModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashInternlmLayer(
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
