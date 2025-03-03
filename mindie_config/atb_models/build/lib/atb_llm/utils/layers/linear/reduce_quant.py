# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from torch import nn


class ReduceQuant(nn.Module):
    def __init__(self, scale: list):
        super().__init__()

        reduce_scale, gather_scale = scale
        self.reduce_quant_scale = nn.Parameter(reduce_scale, requires_grad=False)
        self.gather_quant_scale = nn.Parameter(gather_scale, requires_grad=False)

    @classmethod
    def load(cls, prefix, weights):
        reduce_scale = weights.get_tensor(f"{prefix}.reduce_scale")
        gather_scale = weights.get_tensor(f"{prefix}.gather_scale")
        return cls([reduce_scale, gather_scale])