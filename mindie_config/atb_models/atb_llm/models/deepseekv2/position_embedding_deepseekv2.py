# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch

from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class DeepseekV2YarnRotaryEmbedding(PositionRotaryEmbedding):
    class StaticInputArgs:
        def __init__(self, scaling_factor=1.0, max_position_embeddings=2048, original_max_position_embeddings=4096,
                     beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0,):
            self.scaling_factor = scaling_factor
            self.max_position_embeddings = max_position_embeddings
            self.original_max_position_embeddings = original_max_position_embeddings
            self.beta_fast = beta_fast
            self.beta_slow = beta_slow
            self.mscale = mscale
            self.mscale_all_dim = mscale_all_dim

    def __init__(self, inv_freq, dim, base, scaling_factor=1.0,
                 max_position_embeddings=2048, original_max_position_embeddings=4096,
                 beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0,
                 ):
        super().__init__(inv_freq, scaling_factor)
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_cached_total = None
        self._sin_cached_total = None
        self.mscale = float(
            DeepseekV2YarnRotaryEmbedding.yarn_get_mscale(self.scaling_factor, mscale)
            / DeepseekV2YarnRotaryEmbedding.yarn_get_mscale(self.scaling_factor, mscale_all_dim)
        )

    @staticmethod
    def yarn_find_correction_range(
            low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
    ):
        low = math.floor(
            DeepseekV2YarnRotaryEmbedding.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            DeepseekV2YarnRotaryEmbedding.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
    ):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    @staticmethod
    def yarn_linear_ramp_mask(min_value, max_value, dim):
        if min_value == max_value:
            max_value += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (DeepseekV2YarnRotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (DeepseekV2YarnRotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed

    @classmethod
    def static_yarn(cls, dim, base, device, yarn_kwargs: StaticInputArgs):
        scaling_factor = yarn_kwargs.scaling_factor
        max_position_embeddings = yarn_kwargs.max_position_embeddings
        original_max_position_embeddings = yarn_kwargs.original_max_position_embeddings
        beta_fast = yarn_kwargs.beta_fast
        beta_slow = yarn_kwargs.beta_slow
        mscale = yarn_kwargs.mscale
        mscale_all_dim = yarn_kwargs.mscale_all_dim
        try:
            freq_extra = 1.0 / (
                    base
                    ** (torch.arange(0, dim, 2, device=device, dtype=torch.double) / dim)
            ).to(torch.float)
            freq_inter = 1.0 / (
                    scaling_factor
                    * base
                    ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            low, high = cls.yarn_find_correction_range(
                beta_fast,
                beta_slow,
                dim,
                base,
                original_max_position_embeddings,
            )
            inv_freq_mask = 1.0 - cls.yarn_linear_ramp_mask(low, high, dim // 2).to(
                device=device, dtype=torch.float32
            )
            inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        except ZeroDivisionError as e:
            msg = e.messgae
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ZeroDivisionError from e
        return cls(inv_freq, dim, base, scaling_factor,
                   max_position_embeddings, original_max_position_embeddings,
                   beta_fast, beta_slow, mscale, mscale_all_dim)

    def update_cos_sin_cache(self, dtype, device, seqlen):
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)

    def update_cos_sin_cache_total(self, dtype, device, seqlen):
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached_total.device != device
                or self._cos_cached_total.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached_total = (torch.cos(emb) * self.mscale).to(dtype)
            self._sin_cached_total = (torch.sin(emb) * self.mscale).to(dtype)