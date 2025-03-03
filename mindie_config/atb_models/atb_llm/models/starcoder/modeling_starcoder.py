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


class FlashStarcoderAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_names = [f"{prefix}.c_attn"]
        self.pack_name = f"{prefix}.c_attn"
        self.qkv_bias = True
        self.dense_name = f"{prefix}.c_proj"
        self.dense_bias = True
        layer_prefix = ".".join(self.prefix.split(".")[:-1])
        self.norm_name = f"{layer_prefix}.ln_1"
        self.bias_pre_add = True
        self.load_weights(**kwargs)


class StarcoderMLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.gate_up_names = [f"{prefix}.c_fc"]
        self.gate_up_bias = True
        self.up_weight_only = True
        self.down_name = f"{prefix}.c_proj"
        self.down_bias = True
        layer_prefix = ".".join(self.prefix.split(".")[:-1])
        self.norm_name = f"{layer_prefix}.ln_2"
        self.bias_pre_add = True
        super().load_weights(**kwargs)


class FlashStarcoderLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="transformer", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.prefix = f"{model_prefix}.h.{layer_id}"
        self.norm_bias = True
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashStarcoderAttention(
            prefix=f"{self.prefix}.attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = StarcoderMLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashStarcoderModel(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="transformer", **kwargs):
        super().__init__()
        self.wte = TensorEmbedding(prefix=f"{model_prefix}.wte", weights=weights)
        self.wpe = TensorEmbedding(prefix=f"{model_prefix}.wpe", weights=weights)
        self.layers = torch.nn.ModuleList(
            [
                FlashStarcoderLayer(layer_id, config, weights, model_prefix, **kwargs)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNormBias(prefix=f"{model_prefix}.ln_f", weights=weights, eps=config.layer_norm_epsilon)
