# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass

from atb_llm.models.base.config import BaseConfig


@dataclass
class GPTNeoXConfig(BaseConfig):
    vocab_size: int = 50432
    hidden_size: int = 6144
    num_hidden_layers: int = 44
    num_attention_heads: int = 64
    intermediate_size: int = 24576
    hidden_act: str = "gelu_fast"
    rotary_pct: float = 0.25
    rotary_emb_base: int = 10000
    classifier_dropout: float = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    use_parallel_residual: bool = True
    model_type: str = 'gpt_neox'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'gpt_neox'
