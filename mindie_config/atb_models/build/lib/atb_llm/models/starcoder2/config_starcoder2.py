# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from ..base.config import BaseConfig


@dataclass
class Starcoder2Config(BaseConfig):
    model_type: str = "starcoder2"
    hidden_act: str = "gelu_pytorch_tanh"
    bos_token_id: int = 0
    eos_token_id: int = 0
    hidden_size: int = 6144
    initializer_range: float = 0.01275
    intermediate_size: int = 24576
    num_attention_heads: int = 48
    num_hidden_layers: int = 40
    num_key_value_heads: int = 4
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    mlp_type: str = "default"
    norm_epsilon: float = 1e-05
    norm_type: str = "layer_norm"
    rope_theta: int = 100000
    sliding_window: int = 4096
    use_bias: bool = True
    use_cache: bool = True
    tie_word_embeddings: Optional[bool] = None


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = True
