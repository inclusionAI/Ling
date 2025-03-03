# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
from enum import Enum

import torch
from torch import nn


class NormType(int, Enum):
    RMS_NORM = 0
    LAYER_NORM = 1


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

        return super(FastLayerNorm, self).forward(hidden_states), residual


class RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class RMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=weights.dtype)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class RMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = RMSNorm(prefix, weights, eps)
        self.anti = RMSNormBias(f'{prefix}.module', weights, eps)


class RMSNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = RMSNorm(f'{prefix}.ori', weights, eps)
        self.anti = RMSNormBias(f'{prefix}.anti', weights, eps)
