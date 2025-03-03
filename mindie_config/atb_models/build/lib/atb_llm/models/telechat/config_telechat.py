# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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

""" Telechat configuration"""
from typing import Optional
from ..base.config import BaseConfig


class TelechatConfig(BaseConfig):
    model_type = "telechat"
    vocab_size: int = 160256
    n_layer: int = 30
    n_head: int = 32
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    apply_residual_connection_post_layernorm: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    logn: bool = True
    ffn_hidden_size: int = 12288
    training_seqlen: int = 8192
    embed_layernorm: bool = False
    hidden_act: str = "fastgelu"
    seq_length: int = 4096
    intermediate_size: int = 1

    num_key_value_heads: Optional[int] = None
    rope_given_inv_feq_str: Optional[str] = None
    rope_keep_local_base_windows: Optional[int] = None
    rope_mscale: Optional[int] = None
    rope_vanilla_theta: Optional[float] = None
    
    def __init__(self, **kwargs):
        self.keys_to_ignore_at_inference = ["past_key_values"]
        self.attribute_map = {
            "num_hidden_layers": "n_layer",
            "num_attention_heads": "n_head",
            "num_key_value_heads": "n_head",
            "rms_norm_eps": "layer_norm_epsilon",
            'max_position_embeddings': 'seq_length'
        }
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size: int = 4096 if n_embed is None else n_embed
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'telechat'

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
