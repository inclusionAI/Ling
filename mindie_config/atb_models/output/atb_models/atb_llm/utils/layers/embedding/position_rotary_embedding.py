# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# This file was referenced from project IBM text-generation-inference
import math
from enum import Enum

import torch
from torch import nn
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class PositionEmbeddingType(int, Enum):
    ROPE = 0
    ALIBI = 1
    ABSOLUTE = 2


class PositionRotaryEmbedding(nn.Module):
    def __init__(self, inv_freq, scaling_factor=1.0, base=10000.0):
        super().__init__()

        self.base = base
        self.inv_freq = inv_freq
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_cached_total = None
        self._sin_cached_total = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.scaling_factor = scaling_factor
        self._ntk_alpha_cached = 1.0
        self.position_ids_offset = []
        self.pos_lens = None
        self.ntk_inv_freqs = None
        self.position_ids_expanded = None
        self._batch_seq_len_cached = []
        self._batch_ntk_alpha_cached = []
        self._global_seq_len_cached = {}
        self._global_ntk_alpha_cached = {}
        self._table_size = 0

    @classmethod
    def static(cls, dim, base, device, scaling_factor=1.0):
        try:
            inv_freq = 1.0 / (
                    base
                    ** (torch.arange(0, dim, 2, device=device, dtype=torch.double) / dim)
            ).to(torch.float)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        return cls(inv_freq, scaling_factor, base)

    @classmethod
    def load(cls, prefix, weights):
        # Always load this in float32 !
        dtype = weights.dtype
        weights.dtype = torch.float32
        inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
        weights.dtype = dtype
        return cls(inv_freq)

    def update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            if math.isclose(self.scaling_factor, 0.0):
                raise ValueError("PositionRotaryEmbedding scaling_factor cannot be 0, " \
                    "it is a factor which is recommended to be >= 1.0, check README.md to see how to set rope config.")
            t = t / self.scaling_factor
            # Don't do einsum, it converts fp32 to fp16 #freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def update_cos_sin_cache_total(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached_total.device != device
                or self._cos_cached_total.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            if math.isclose(self.scaling_factor, 0.0):
                raise ValueError("PositionRotaryEmbedding scaling_factor cannot be 0, " \
                    "it is a factor which is recommended to be >= 1.0, check README.md to see how to set rope config.")
            t = t / self.scaling_factor
            # Don't do einsum, it converts fp32 to fp16 # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached_total = torch.cos(emb).to(dtype)
            self._sin_cached_total = torch.sin(emb).to(dtype)

    def update_llama3_cos_sin_cache_total(self, config, dtype, device, seqlen):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached_total.device != device
            or self._cos_cached_total.dtype != dtype
        ):
            validate_llama3_freq_factors(config.rope_scaling)
            factor = config.rope_scaling.factor
            low_freq_factor = config.rope_scaling.low_freq_factor
            high_freq_factor = config.rope_scaling.high_freq_factor
            old_context_len = config.rope_scaling.original_max_position_embeddings

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / self.inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

            position_ids_total = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(inv_freq_llama.to(position_ids_total.device), position_ids_total).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached_total = torch.cos(emb).to(dtype)
            self._sin_cached_total = torch.sin(emb).to(dtype)
    
    def clear_ntk_cache(self, batch_size):
        self._batch_seq_len_cached = [0 for i in range(batch_size)]
        self._batch_ntk_alpha_cached = [1.0 for i in range(batch_size)]
        self.position_ids_offset = [0 for i in range(batch_size)]
        self.ntk_inv_freqs = None
        self.pos_lens = None
        self.position_ids_expanded = None
        self._global_seq_len_cached = {}
        self._global_ntk_alpha_cached = {}
        self._table_size = 0

    def set_ntk_cache(self, seq_len, inv_freq, device):
        self._seq_len_cached = seq_len
        self.ntk_inv_freqs = inv_freq[None, :].to(device)
        pos_len = torch.tensor([seq_len], dtype=torch.int32, device=device)
        position_ids_expanded = torch.arange(seq_len, dtype=torch.int32, device=device)
        self.pos_lens = pos_len
        self.position_ids_expanded = position_ids_expanded
    
    def dynamic_ntk_inv_freq_llama(self, config, seqlen, device, ntk_alpha):
        base = self.base
        dim = config.hidden_size // config.num_attention_heads
        max_position_embeddings = config.max_position_embeddings
        if seqlen > self._seq_len_cached:
            base = base * ntk_alpha ** (dim / (dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            self.set_ntk_cache(seqlen, inv_freq, device)

        if seqlen < max_position_embeddings and self._seq_len_cached > max_position_embeddings:
            self.set_ntk_cache(max_position_embeddings, self.inv_freq, device)
    
    def dynamic_ntk_inv_freq(self, config, seq_len, device, ntk_alpha, req_id):
        base = config.rotary_emb_base
        dim = config.hidden_size // config.num_attention_heads
        if ntk_alpha in self._global_ntk_alpha_cached and seq_len < self._global_ntk_alpha_cached[ntk_alpha]:
            self.position_ids_offset[req_id] = self._global_ntk_alpha_cached[ntk_alpha]
            self._batch_seq_len_cached[req_id] = self._global_seq_len_cached.get(ntk_alpha, 0)
            self._batch_ntk_alpha_cached[req_id] = ntk_alpha
        elif seq_len > self._batch_seq_len_cached[req_id] or ntk_alpha != self._batch_ntk_alpha_cached[req_id]:
            base = base * ntk_alpha ** (dim / (dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            inv_freq = inv_freq[None, :].float()
            self.ntk_inv_freqs = inv_freq if self.ntk_inv_freqs is None \
                                else torch.cat([self.ntk_inv_freqs, inv_freq], dim=0)

            seq_len_cached = max(2 * seq_len, 16)
            self._batch_seq_len_cached[req_id] = seq_len_cached
            self._batch_ntk_alpha_cached[req_id] = ntk_alpha

            pos_len = torch.tensor([seq_len_cached], dtype=torch.int32, device=device)
            self.pos_lens = pos_len if self.pos_lens is None else torch.cat([self.pos_lens, pos_len], dim=0)
            
            position_ids_expanded = torch.arange(seq_len_cached, dtype=torch.int32, device=device)
            self.position_ids_expanded = position_ids_expanded \
                                        if self.position_ids_expanded is None \
                                        else torch.cat([self.position_ids_expanded, position_ids_expanded], dim=0)
            self.position_ids_offset[req_id] = self._table_size
            self._table_size += seq_len_cached
            self._global_ntk_alpha_cached[ntk_alpha] = self.position_ids_offset[req_id]
            self._global_seq_len_cached[ntk_alpha] = self._batch_seq_len_cached[req_id]

    def yarn_scaling_rotary_embedding(self, config, device, seqlen):
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        max_position_embeddings = config.max_position_embeddings if \
                                    config.rope_scaling.original_max_position_embeddings is None \
                                    else config.rope_scaling.original_max_position_embeddings
        factor = 8.0 if config.rope_scaling.factor is None else config.rope_scaling.factor
        
        beta_fast = config.rope_scaling.beta_fast
        beta_slow = config.rope_scaling.beta_slow

        def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
        
        def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
            return max(low, 0), min(high, dim - 1)
        
        def linear_ramp_mask(low, high, dim):
            if low == high:
                high += 0.001

            linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func
            
        if seqlen > self._seq_len_cached:    
            pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (factor * pos_freqs)

            low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

            inv_freq_mask = 1 - linear_ramp_mask(low, high, dim // 2).float().to(device)
            inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
            self.set_ntk_cache(seqlen, inv_freq, device)

    def get_cos_sin(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
    ):
        """
        Return cos and sin for the asked position ids
        """

        self.update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.nn.functional.embedding(position_ids, self._cos_cached)
        sin = torch.nn.functional.embedding(position_ids, self._sin_cached)

        return cos.unsqueeze(1), sin.unsqueeze(1)

    def get_cos_sin_total(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
    ):
        """
        Return cos and sin for the asked position ids
        """

        self.update_cos_sin_cache_total(dtype, position_ids.device, max_s)

        cos = torch.nn.functional.embedding(position_ids, self._cos_cached_total)
        sin = torch.nn.functional.embedding(position_ids, self._sin_cached_total)

        return cos, sin

    def get_cos_cached_total(self):
        return self._cos_cached_total

    def get_sin_cached_total(self):
        return self._sin_cached_total

    def get_cos_sin_cached_total(self, position_ids):
        cos = torch.index_select(self._cos_cached_total, 0, position_ids)
        sin = torch.index_select(self._sin_cached_total, 0, position_ids)
        return cos, sin

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rotary_dim = cos.shape[-1]
        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim: 2 * rotary_dim]
        x_rope = x[..., :2 * rotary_dim]

        # do original forward
        dtype = x.dtype
        cos_compute = torch.cat((cos, cos), dim=-1)
        sin_compute = torch.cat((sin, sin), dim=-1)
        x_rope = (x_rope * cos_compute) + (torch.cat((-x2, x1), dim=-1) * sin_compute)
        x[..., :2 * rotary_dim] = x_rope
        x = x.to(dtype)
        return x


def validate_llama3_freq_factors(rope_config):
    if rope_config.low_freq_factor is None or rope_config.low_freq_factor <= 0.0:
        error_msg = f"Invalid rope_scaling.low_freq_factor value: {rope_config.low_freq_factor}. " \
            "This field must be a float greater than 0.0 if rope_scaling.rope_type is `llama3`. " \
            "Please set it a valid value in the config.json -> rope_scaling -> low_freq_factor, " \
            "as a reference, it is 1.0 in the original llama3 implementation."
        logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError(error_msg)
    if rope_config.high_freq_factor is None or rope_config.high_freq_factor <= rope_config.low_freq_factor:
        error_msg = f"Invalid rope_scaling.high_freq_factor value: {rope_config.high_freq_factor}, " \
            f"it must be a float greater than rope_scaling.low_freq_factor = {rope_config.low_freq_factor}. " \
            "Please set it a valid value in the config.json -> rope_scaling -> high_freq_factor, " \
            "as a reference, (low_freq_factor, high_freq_factor) is (1.0, 4.0) in the original llama3 implementation."
        logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError(error_msg)

    