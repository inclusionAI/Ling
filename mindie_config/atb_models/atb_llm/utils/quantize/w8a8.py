# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .quant_type import LinearTypeV2


class W8A8LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None, input_offset=None, bias=None):
        super().__init__()
        super(nn.Module, self).__init__()
        self.linear_desc = LinearTypeV2.W8A8

        self.register_buffer('weight', weight.to(torch.int8))

        self.act_quant_name = 'per_tensor'
        self.register_buffer('input_scale', input_scale)

        if input_offset is not None:
            self.register_buffer('input_offset', input_offset.to(torch.int8))
        else:
            self.register_buffer('input_offset', torch.tensor([], dtype=torch.int8))

        self.weight_quant_name = 'per_channel'

        self.register_buffer('deq_scale', deq_scale)

        if quant_bias is not None:
            self.register_buffer('quant_bias', quant_bias)
            self.has_bias = True
        else:
            self.quant_bias = None

        self.output_quant_name = 'per_channel'

        if bias is not None:
            self.register_buffer('bias', bias)
