# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from atb_llm.models.base.config import BaseConfig


@dataclass
class InternlmConfig(BaseConfig):
    vocab_size: int = 103168
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    bias: bool = True
    rope_theta: int = 10000
    rope_scaling: Optional[float] = None
    attn_implementation: str = "eager"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'internlm'
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
