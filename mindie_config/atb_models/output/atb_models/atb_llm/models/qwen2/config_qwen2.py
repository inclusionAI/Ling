# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass

from ..base.config import BaseConfig


@dataclass
class Qwen2Config(BaseConfig):
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    model_type = "qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "tie_word_embeddings" in kwargs:
            self.tie_word_embeddings = kwargs.get("tie_word_embeddings")
        if "num_key_value_heads" not in kwargs:
            self.num_key_value_heads = self.num_attention_heads
