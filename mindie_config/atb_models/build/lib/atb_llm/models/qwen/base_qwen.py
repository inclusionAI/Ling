# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
import importlib
from typing import Optional, List, Tuple

import torch
import torch_npu
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.env import ENV
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log import logger, print_log


from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType


class QwenRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq_custom = None
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if importlib.util.find_spec("einops") is None:
            msg = "einops is required for Rotary Embedding"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise RuntimeError(msg)

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        self._ntk_alpha_cached_list = [1.0]
    
    def get_ntk_alpha_list(self):
        return self._ntk_alpha_cached_list
    
    def set_ntk_alpha_list(self, ntk_alpha_list):
        self._ntk_alpha_cached_list = ntk_alpha_list

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq_custom = 1.0 / (
                    base
                    ** (
                            torch.arange(0, self.dim, 2, device=self.inv_freq_custom.device).float()
                            / self.dim
                    )
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq_custom.device)
            freqs = torch.outer(seq.type_as(self.inv_freq_custom), self.inv_freq_custom)

            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            emb = rearrange(emb, "n d -> 1 n 1 d")

            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        cos, sin = self._rotary_pos_emb_cache
        return [cos[:, offset: offset + max_seq_len], sin[:, offset: offset + max_seq_len]]
