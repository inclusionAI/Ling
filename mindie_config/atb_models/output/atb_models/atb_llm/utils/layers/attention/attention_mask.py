# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch

from torch import nn


class AttentionMask(nn.Module):
    def __init__(self, atten_mask):
        super().__init__()
        self._seq_len_cached = 0
        self.atten_mask_cache = atten_mask

    @classmethod
    def static(cls, max_seq_len, dtype=torch.float16):
        bias_cache = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)).view(max_seq_len,
                                                                                               max_seq_len)
        bias_cache = ~bias_cache
        if dtype == torch.float16:
            mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
        attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)), bias_cache, mask_value)
        return cls(attn_mask)

    def get_decode_attn_mask(
        self, input_lengths:torch.tensor, max_s: int, dtype: torch.dtype, device: torch.device
    ):
        bs = input_lengths.shape[0]
        attn_mask = torch.ones((bs, max_s), dtype=dtype).to(device)
        input_lengths_unsqueeze = input_lengths.unsqueeze(1)
        token_index = torch.arange(0, max_s).repeat(bs).view(bs, max_s).to(device)
        attn_mask[token_index < input_lengths_unsqueeze] = 0
        return attn_mask.view(-1, 1, max_s)

    def update_attn_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self.atten_mask_cache.dtype != dtype:
            self._seq_len_cached = seqlen
            bias_cache = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool)).view(seqlen, seqlen)
            bias_cache = ~bias_cache
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            mask_atten_cache = torch.masked_fill(torch.zeros(size=(seqlen, seqlen)), bias_cache, mask_value)
            self.atten_mask_cache = mask_atten_cache.to(dtype)
        if self.atten_mask_cache.device != device:
            self.atten_mask_cache = self.atten_mask_cache.to(device)

    def get_attn_mask(
            self, max_s: int, dtype: torch.dtype, device: torch.device
    ):
        self.update_attn_cache(dtype, device, max_s)
        return self.atten_mask_cache[:max_s, :max_s]
