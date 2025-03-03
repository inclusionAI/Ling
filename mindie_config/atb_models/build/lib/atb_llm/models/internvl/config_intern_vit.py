# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from atb_llm.models.base.config import BaseConfig


@dataclass
class InternVisionConfig(BaseConfig):
    num_channels: int = 3
    patch_size: int = 14
    image_size: int = 224
    qkv_bias: bool = False
    hidden_size: int = 3200
    num_attention_heads: int = 25
    intermediate_size: int = 12800
    qk_normalization: bool = True
    num_hidden_layers: int = 48
    use_flash_attn: bool = True
    hidden_act: str = "gelu"
    norm_type: str = "rms_norm"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'intern_vit_6b'
