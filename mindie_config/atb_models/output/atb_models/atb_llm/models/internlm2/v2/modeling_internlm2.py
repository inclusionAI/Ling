# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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


class Internlm2MLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.gate_up_names = [f'{prefix}.w1', f'{prefix}.w3']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.ffn_norm'
        self.pack_name = f'{prefix}.w1_w3'
        self.down_name = f'{prefix}.w2'
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
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            gate_up_linear = TensorParallelColumnLinear.load(
                self.config,
                prefix=self.pack_name,
                weights=self.weights,
                bias=False,
            )
            gate_up_linear.linear.num_linear_before_pack = 2
            gate_up_linear.linear.prefixes = self.gate_up_names
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


class FlashInternlm2Attention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_names = [f'{prefix}.wqkv']
        self.pack_name = f'{prefix}.wqkv'
        self.dense_name = f'{prefix}.wo'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.attention_norm'
        self.load_weights(**kwargs)

    def load_qkv_weights(self, **kwargs):
        query_key_value_linear = TensorParallelColumnLinear.load(
            self.config,
            prefix=self.pack_name,
            weights=self.weights,
            bias=False,
        )
        query_key_value_linear.linear.num_linear_before_pack = 3
        setattr(self, get_suffix(self.pack_name), query_key_value_linear)


class FlashInternlm2Layer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="model", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.prefix = f"{model_prefix}.layers.{layer_id}"
        self.attn_name = "attention"
        self.mlp_name = "feed_forward"
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.attention = FlashInternlm2Attention(
            prefix=f"{self.prefix}.attention", config=self.config, weights=self.weights, **kwargs
        )
        self.feed_forward = Internlm2MLP(
            prefix=f"{self.prefix}.feed_forward", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)

    def load_input_layernorm_weight(self, **kwargs):
        self.norm_name = f"{self.prefix}.attention_norm"
        super().load_input_layernorm_weight(**kwargs)

    def load_post_attention_layernorm_weight(self, **kwargs):
        self.norm_name = f"{self.prefix}.ffn_norm"
        super().load_post_attention_layernorm_weight(**kwargs)


class FlashInternlm2Model(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="model"):
        super().__init__()

        self.is_embedding_parallel = config.vocab_size > 100000
        self.tok_embeddings = TensorEmbedding(
            prefix=f"{model_prefix}.tok_embeddings", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashInternlm2Layer(
                    layer_id,
                    config,
                    weights,
                    model_prefix,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )
