# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from ..base.config import BaseConfig


@dataclass
class StarcoderConfig(BaseConfig):
    model_type: str = "starcoder"
    hidden_act: str = "gelu"
    hidden_size: int = 6144
    bos_token_id: int = 0
    eos_token_id: int = 0
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05
    intermediate_size: int = 24576
    num_attention_heads: int = 48
    num_hidden_layers: int = 40
    attention_softmax_in_fp32: bool = True
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    inference_runner: int = 0
    pad_key_length: bool = True
    pre_allocate_kv_cache: bool = False
    resid_pdrop: float = 0.1
    scale_attention_softmax_in_fp32: bool = True
    scale_attn_weights: bool = True
    summary_first_dropout: float = 0.1
    summary_proj_to_labels: bool = True
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    use_cache: bool = True
    validate_runner_input: bool = True
    max_batch_size: Optional[int] = None
    max_sequence_length: Optional[int] = None
    summary_activation: Optional[str] = None

    def __init__(self, **kwargs):
        self.attribute_map = {
            'activation_function': 'hidden_act',
            'n_embd': 'hidden_size',
            'n_head': 'num_attention_heads',
            'n_inner': 'intermediate_size',
            'n_positions': 'max_position_embeddings',
            'n_layer': 'num_hidden_layers',
        }
        super().__init__(**kwargs)
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if 'num_key_value_heads' not in kwargs :
            self.num_key_value_heads = 1
