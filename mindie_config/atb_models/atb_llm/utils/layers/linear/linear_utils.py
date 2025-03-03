# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from collections import OrderedDict

import torch
from torch import nn
import torch_npu

from atb_llm.utils.env import ENV
from ....utils.initial import NPUSocInfo
from ....utils.quantize.pack_type import TransposeType
from ....utils.quantize.quant_type import LinearTypeV2


class LinearUtils:
    soc_info = None

    def __init__(self):
        self.prefix = None
        self.linear_desc = LinearTypeV2.INVALID
        self.trans_flag = TransposeType.TRANSPOSE
        self.has_bias = False
        if not LinearUtils.soc_info:
            LinearUtils.set_soc_info()
        self.prefixes = []
        self.num_linear_before_pack = 1
        self.tensor_parallel_dim = 0
        self.align_size = 1

    @classmethod
    def set_soc_info(cls):
        cls.soc_info = NPUSocInfo()
    
    @classmethod
    def weight_format_cast(cls, tensor):
        if not cls.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    def check_transpose(self):
        if self.soc_info.need_nz or not ENV.auto_transpose_enable:
            return TransposeType.TRANSPOSE

        is_k_divisible = self.weight.shape[1] % 256 == 0
        is_n_divisible = self.weight.shape[0] % 256 == 0
        if not is_k_divisible and is_n_divisible:
            return TransposeType.NOT_TRANSPOSE
        return TransposeType.TRANSPOSE

    def transpose_weight_as_need(self):
        self.trans_flag = self.check_transpose()
        if not self.trans_flag:
            self.weight = nn.Parameter(torch.transpose(self.weight, 0, 1).contiguous())

    def get_weights(self, prefix):
        self.prefix = prefix
        weights_dict = OrderedDict()
        for name, buf in self.named_buffers():
            weights_dict[f"{prefix}.{name}"] = self.weight_format_cast(buf.data)
        return weights_dict
