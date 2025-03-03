# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional
from ..base.config import BaseConfig


@dataclass
class Qwen2MoeConfig(BaseConfig):
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
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
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 1408
    shared_expert_intermediate_size: int = 5632
    num_experts_per_tok: int = 4
    num_experts: int = 60
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: Optional[list] = None
    tp: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'qwen2_moe'
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        else:
            self.mlp_only_layers = self.mlp_only_layers