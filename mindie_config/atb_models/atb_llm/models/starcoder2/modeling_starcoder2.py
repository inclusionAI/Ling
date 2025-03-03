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
from atb_llm.utils.layers import TensorEmbedding, RMSNormBias
from atb_llm.models.base.modeling import FlashAttention, MLP, FlashLayer


class FlashStarcoder2Attention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_bias = True
        self.dense_bias = True
        self.bias_pre_add = True
        self.load_weights(**kwargs)


class Starcoder2MLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.gate_up_names = [f"{prefix}.c_fc"]
        self.gate_up_bias = True
        self.up_weight_only = True
        self.down_name = f"{prefix}.c_proj"
        self.down_bias = True
        self.bias_pre_add = True
        super().load_weights(**kwargs)


class FlashStarcoder2Layer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="model", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.norm_bias = True
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashStarcoder2Attention(
            prefix=f"{self.prefix}.self_attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = Starcoder2MLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashStarcoder2Model(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="model", **kwargs):
        super().__init__()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = torch.nn.ModuleList(
            [
                FlashStarcoder2Layer(layer_id, config, weights, model_prefix, **kwargs)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNormBias(prefix=f"{model_prefix}.norm", weights=weights, eps=config.norm_epsilon)
