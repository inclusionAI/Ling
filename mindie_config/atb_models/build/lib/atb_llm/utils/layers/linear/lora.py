# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from torch import nn


class Lora(nn.Module):
    def __init__(self, lora_a, lora_b, r, alpha):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.lora_a = nn.Parameter(lora_a)
        self.lora_b = nn.Parameter(lora_b)
