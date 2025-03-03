# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass

from ..base.config import BaseConfig


@dataclass
class Glm4vConfig(BaseConfig):
    hidden_act: str = "silu"
    intermediate_size: int = 11008

    def __init__(self, **kwargs):
        self.attribute_map = {
            'seq_length': 'max_position_embeddings',
            'padded_vocab_size': 'vocab_size',
            'num_layers': 'num_hidden_layers'
        }
        super().__init__(**kwargs)
        self.model_type = 'glm4v'
        if 'llm_model_type' not in kwargs:
            self.llm_model_type = 'chatglm'
        if 'num_key_value_heads' not in kwargs:
            self.num_key_value_heads = self.multi_query_group_num
        if 'rope_ratio' not in kwargs:
            self.rope_ratio = 1
