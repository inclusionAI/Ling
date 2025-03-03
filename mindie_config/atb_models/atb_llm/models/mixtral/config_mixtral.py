# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional
import torch
from ..base.config import BaseConfig


@dataclass
class MixtralConfig(BaseConfig):
    model_type: str = "mixtral"
    attention_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_experts_per_tok: int = 2
    num_hidden_layers: int = 32
    num_key_value_heads: int = 8
    n_routed_experts: int = 8
    output_router_logits: bool = False
    rms_norm_eps: float = 1e-05
    rope_theta: float = 1000000.0
    router_aux_loss_coef: float = 0.02
    tie_word_embeddings: bool = False
    use_cache: bool = True
    sliding_window: Optional[int] = None

    def __init__(self, **kwargs):
        self.attribute_map = {'num_local_experts': 'n_routed_experts'}
        super().__init__(**kwargs)
        if 'tp' not in kwargs:
            self.tp = True
        self.torch_dtype = torch.float16 # 暂不支持bf16
