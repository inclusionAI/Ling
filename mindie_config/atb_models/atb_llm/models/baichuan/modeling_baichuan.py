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
from torch import nn

from atb_llm.models.base.modeling import FlashAttention, MLP, FlashLayer
from atb_llm.utils.layers import (
    TensorEmbedding,
    TensorParallelEmbedding,
    RMSNorm
)


class BaichuanMLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)


class FlashBaichuanAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_names = [f'{self.prefix}.W_pack']
        self.pack_name = f'{self.prefix}.W_pack'
        self.load_weights(**kwargs)


class FlashBaichuanLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="model", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashBaichuanAttention(
            prefix=f"{self.prefix}.self_attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = BaichuanMLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashBaichuanModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.parallel_embedding = config.vocab_size == 125696
        self.embed_tokens = (TensorParallelEmbedding if self.parallel_embedding else TensorEmbedding)(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashBaichuanLayer(layer_id, config, weights) for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
