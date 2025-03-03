# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass

from ..base.config import BaseConfig


@dataclass
class BloomConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.seq_length = 4096
        self.attribute_map = {
            'max_position_embeddings': 'seq_length',
            'num_hidden_layers': 'n_layer',
            'n_head': 'num_attention_heads',
            'hidden_size': 'n_embed',
        }
        super().__init__(**kwargs)
