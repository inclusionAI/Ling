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

from torch import nn
from atb_llm.models.base.modeling import FlashAttention, FlashLayer, MLP
from atb_llm.utils.layers import TensorParallelRowLinear, KvCache, RMSNorm
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type


class FlashChatglmAttention(FlashAttention):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__(prefix, config, weights)
        self.qkv_names = [f'{prefix}.query_key_value']
        self.qkv_bias = True
        self.kv_head_nums_per_rank = max(config.multi_query_group_num // weights.process_group.size(), 1)

        if config.quantization_config.kv_quant_type is not None:
            self.kv_cache_quant = KvCache.load(prefix_k=f"{prefix}.query_key_value.k_proj",
                                               prefix_v=f"{prefix}.query_key_value.v_proj",
                                               weights=weights,
                                               gqa_size=self.kv_head_nums_per_rank * config.kv_channels)

        self.pack_type = calc_linear_pack_type(weights, self.qkv_names, self.norm_name)
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=False,
        )
        
        self.load_qkv_weights()


class ChatglmMLP(MLP):
    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)

        self.gate_up_names = [f'{prefix}.dense_h_to_4h']
        self.pack_name = f'{prefix}.dense_h_to_4h'
        self.down_name = f"{prefix}.dense_4h_to_h"
        self.load_weights()


class FlashChatglmLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, prefix="transformer.encoder"):
        super().__init__(layer_id, config, weights, prefix)

        self.attn_name = "self_attention"
        self.config.rms_norm_eps = config.layernorm_epsilon

        self.self_attention = FlashChatglmAttention(
            prefix=f"{self.prefix}.{self.attn_name}", config=config, weights=weights
        )
        self.mlp = ChatglmMLP(prefix=f"{self.prefix}.{self.mlp_name}", config=config, weights=weights)
        self.load_weights()


class GLMTransformer(nn.Module):
    def __init__(self, config, weights):
        super(GLMTransformer, self).__init__()

        if config.quantize == "w8a8sc":
            prefix = "encoder"
        else:
            prefix = "transformer.encoder"

        self.layers = nn.ModuleList(
            [
                FlashChatglmLayer(layer_id, config, weights, prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm = RMSNorm(
            prefix=f"{prefix}.final_layernorm", weights=weights, eps=config.layernorm_epsilon
        )
