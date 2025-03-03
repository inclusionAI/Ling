# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass

from ..base.config import BaseConfig


@dataclass
class QwenConfig(BaseConfig):
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    emb_dropout_prob: float = 0.0
    attn_dropout_prob: float = 0.0
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    max_position_embeddings: int = 8192
    scale_attn_weights: bool = True
    use_cache: bool = True
    bf16: bool = False
    fp16: bool = False
    fp32: bool = False
    kv_channels: int = 128
    rotary_pct: float = 1.0
    rotary_emb_base: int = 10000
    use_dynamic_ntk: bool = True
    use_logn_attn: bool = True
    use_flash_attn: str = "auto"
    intermediate_size: int = 22016
    no_bias: bool = True
    tie_word_embeddings: bool = False
    use_cache_quantization: bool = False
    use_cache_kernel: bool = False
    softmax_in_fp32: bool = False
    model_type = "qwen"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_act = "silu"
        if "tie_word_embeddings" in kwargs:
            self.tie_word_embeddings = kwargs.get("tie_word_embeddings")
        if "num_key_value_heads" not in kwargs:
            self.num_key_value_heads = self.num_attention_heads
