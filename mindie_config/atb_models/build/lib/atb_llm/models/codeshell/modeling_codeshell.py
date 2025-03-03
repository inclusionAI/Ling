# Copyright 2022 EleutherAI and the HuggingFace Inc. team
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

from torch import nn
from atb_llm.models.base.modeling import FlashAttention, MLP, FlashLayer, get_suffix
from atb_llm.utils.layers import (
    RMSNormBias,
    TensorParallelRowLinear
)


class FlashCodeshellAttention(FlashAttention):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__(prefix, config, weights)
        self.num_kv_heads = config.multi_query_group_num
        self.qkv_names = [f'{prefix}.c_attn']
        self.pack_name = f'{prefix}.c_attn'
        self.qkv_bias = True
        self.dense_name = f'{prefix}.c_proj'
        self.dense_bias = True
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.ln_1'
        self.bias_pre_add = True
        self.load_weights()

    def load_dense_weights(self, **kwargs):
        dense_linear = TensorParallelRowLinear.load(
            self.config,
            prefix=self.dense_name,
            weights=self.weights,
            bias=self.dense_bias,
            bias_pre_add=self.bias_pre_add
        )
        setattr(self, get_suffix(self.dense_name), dense_linear)


class CodeshellMLP(MLP):
    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)

        self.gate_up_names = [f'{prefix}.c_fc']
        self.up_weight_only = True
        self.gate_up_bias = True
        self.down_name = f'{prefix}.c_proj'
        self.down_bias = True
        self.bias_pre_add = True
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.ln_2'
        self.load_weights()


class FlashCodeshellLayer(FlashLayer):
    def __init__(self, layer_id, config, weights):
        super().__init__(layer_id, config, weights)
        self.prefix = f"transformer.h.{layer_id}"
        self.attn_name = "attn"
        self.norm_bias = True
        self.load_weights()

    def load_weights(self, **kwargs):
        self.attn = FlashCodeshellAttention(
            prefix=f"{self.prefix}.attn", config=self.config, weights=self.weights
        )
        self.mlp = CodeshellMLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights)
        super().load_weights(**kwargs)


class GPTTransformer(nn.Module):
    def __init__(self, config, weights):
        super(GPTTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                FlashCodeshellLayer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm = RMSNormBias(
            prefix="transformer.ln_f", weights=weights, eps=config.layer_norm_epsilon
        )
