# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from ..base.flash_causal_multimodal import MultiModalConfig
from ..qwen2.config_qwen2 import Qwen2Config


@dataclass
class Qwen2vlConfig(MultiModalConfig):
    model_type = "qwen2_vl"

    def __init__(
            self,
            vocab_size=152064,
            hidden_size=8192,
            intermediate_size=29568,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=1000000.0,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=80,
            attention_dropout=0.0,
            vision_config=None,
            rope_scaling=None,
            **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen2VLVisionConfig()

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.mrope_section = rope_scaling
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        text_config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
        )

        super().__init__(vision_config=vision_config, text_config=text_config, **kwargs)
