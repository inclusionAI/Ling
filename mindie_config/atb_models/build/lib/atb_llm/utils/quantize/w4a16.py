# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2


def int42int8(weight):
    weight = weight.to(torch.int8)
    k, n = weight.shape
    n_new = n // 2 + n % 2

    if n_new != n // 2:
        raise AssertionError("n dimension should be even")

    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
    weight_res = weight_add.reshape(k, n_new)
    return weight_res


class W4A16LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, weight_scale, weight_offset, bias=None):
        super().__init__()
        super(nn.Module, self).__init__()

        self.weight_quant_name = 'w4a16'

        self.trans_flag = TransposeType.NOT_TRANSPOSE
        self.linear_desc = LinearTypeV2.W4A16

        weight_trans = weight.T.contiguous()  # k, n
        weight_compact = int42int8(weight_trans)  # k, n // 2
        self.register_buffer('weight', weight_compact.to(torch.int8))

        self.register_buffer('weight_scale', weight_scale.T.contiguous())  # 1, n

        if weight_offset is not None:
            self.register_buffer('weight_offset', (-weight_offset).T.contiguous())  # 1, n
        else:
            self.weight_offset = None

        if bias is not None:
            if bias.dtype == torch.bfloat16:
                bias = bias.to(torch.float32)
            self.register_buffer('bias', bias)
            self.has_bias = True
