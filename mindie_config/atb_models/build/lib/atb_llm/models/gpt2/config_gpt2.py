# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from ..base.config import BaseConfig


@dataclass
class Gpt2Config(BaseConfig):
    model_type: str = "gpt2"
    activation_function: str = "gelu_new"
    n_embd: int = 768
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05
    n_inner: int = 24576
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    
    resid_pdrop: float = 0.1
    
    summary_first_dropout: float = 0.1
    summary_proj_to_labels: bool = True
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: Optional[str] = None
    use_cache: bool = True
    validate_runner_input: bool = True
    

    def __init__(self, **kwargs):
        self.attribute_map = {
            'hidden_act': 'activation_function',
            'hidden_size': 'n_embd',
            'num_attention_heads': 'n_head',
            'intermediate_size': 'n_inner',
            'max_position_embeddings': 'n_positions',
            'num_hidden_layers': 'n_layer',
        }
        super().__init__(**kwargs)
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if 'num_key_value_heads' not in kwargs:
            self.num_key_value_heads = 1
