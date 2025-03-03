# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional
import torch
from ..base.config import BaseConfig


@dataclass
class DeepseekConfig(BaseConfig):
    model_type: str = "deepseek"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    aux_loss_alpha: float = 0.001
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    first_k_dense_replace: int = 1
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 10944
    max_position_embeddings: int = 4096
    moe_intermediate_size: int = 1408
    moe_layer_freq: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    norm_topk_prob: bool = False
    num_attention_heads: int = 16
    num_experts_per_tok: int = 6
    num_hidden_layers: int = 28
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[int] = None
    rope_theta: float = 10000.0
    scoring_func: str = "softmax"
    seq_aux: bool = True
    tie_word_embedding: bool = False
    use_cache: bool = True
    vocab_size: int = 102400


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'world_size' not in kwargs:
            self.world_size = 8
        if 'tp' not in kwargs:
            self.tp = True
        self.torch_dtype = torch.bfloat16 # 暂不支持bf16
