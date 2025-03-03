# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from ...base.config import BaseConfig


@dataclass
class BaichuanConfig(BaseConfig):
    vocab_size: int = 125696
    hidden_size: int = 5120
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 40
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    gradient_checkpointing: bool = False
    z_loss_weight: int = 0
    model_type = 'baichuan'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_key_value_heads = self.num_attention_heads