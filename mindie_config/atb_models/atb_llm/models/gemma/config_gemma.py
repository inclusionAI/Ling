# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional
from ..base.config import BaseConfig


@dataclass
class GemmaConfig(BaseConfig):
    alibi_bias_max: float = 8.0
    hidden_act: str = "gelu"
    hidden_size: int = 3072
    initializer_range: float = 0.02
    intermediate_size: int = 24576
    num_attention_heads: int = 16
    num_hidden_layers: int = 28

    pe_type: str = "ROPE"
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    num_key_value_heads: Optional[int] = None
    use_cache: bool = True

    rope_given_inv_feq_str: Optional[str] = None
    rope_keep_local_base_windows: Optional[int] = None
    rope_mscale: Optional[int] = None
    rope_vanilla_theta: Optional[float] = None

    def __init__(self, **kwargs):
        self.attribute_map = {'max_sequence_length': 'max_position_embeddings'}
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'gemma'
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads