# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
""" Shell configuration"""

from dataclasses import dataclass
from ..base.config import BaseConfig


@dataclass
class CodeshellConfig(BaseConfig):
    pad_token_id: int = 0
    bos_token_id: int = 70000
    eos_token_id: int = 70000
    kv_channels: int = 128

    def __init__(self, **kwargs):
        self.attribute_map = {
                                'max_position_embeddings': 'n_positions',
                                'hidden_size': 'n_embd',
                                'num_hidden_layers': 'n_layer',
                                'num_attention_heads': 'n_head',
                                'hidden_act': 'activation_function',
                                'multi_query_group_num': 'num_query_groups',
                                'num_key_value_heads': 'num_query_groups'
        }
        super().__init__(**kwargs)
        if 'n_inner' in kwargs:
            self.n_inner = kwargs['n_inner']
