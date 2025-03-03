# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2


class W8A16LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, weight_scale, weight_offset, bias=None):
        super().__init__()
        super(nn.Module, self).__init__()
        self.weight_quant_name = 'w8a16'

        self.trans_flag = TransposeType.TRANSPOSE
        self.linear_desc = LinearTypeV2.W8A16

        self.register_buffer('weight', weight.to(torch.int8)
                             if self.trans_flag == TransposeType.TRANSPOSE else weight.T.contiguous().to(torch.int8))

        self.register_buffer('weight_scale', weight_scale
                             if self.trans_flag == TransposeType.TRANSPOSE else weight_scale.T.contiguous())

        if weight_offset is not None:
            self.register_buffer(
                'weight_offset',
                -(weight_offset if self.trans_flag == TransposeType.TRANSPOSE else weight_offset.T.contiguous())
            )
        else:
            self.weight_offset = None

        if bias is not None:
            if bias.dtype == torch.bfloat16:
                bias = bias.to(torch.float32)
            self.register_buffer('bias', bias)
            self.has_bias = True
