# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Part of this file was copied from project llmss 0.1.0
from collections import OrderedDict

import torch
from torch import nn
from torch.functional import F

from .linear_utils import LinearUtils
from ....utils.log import logger
from ....utils.quantize.quant_type import LinearTypeV2


class FastLinear(nn.Module, LinearUtils):
    def __init__(
            self,
            weight,
            bias,
            is_norm=False,
    ) -> None:
        super().__init__()
        super(nn.Module, self).__init__()
        if not isinstance(weight, torch.Tensor) or weight.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("linear type not matched, please check `config.json` `quantize` parameter")
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(weight.dtype))
            self.has_bias = True
        else:
            self.bias = None

        self.transpose_weight_as_need()
        self.linear_desc = LinearTypeV2.FLOAT16 if weight.dtype == torch.float16 else LinearTypeV2.BFLOAT16
        self.is_norm_head = is_norm
        self.first_flag = True

    @classmethod
    def load(cls, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.is_norm_head:
            if self.first_flag:
                self.first_flag = False
                self.weight = nn.Parameter(F.normalize(self.weight))
                logger.info("do normalize weight for norm head")
            return F.linear(input_tensor, self.weight, self.bias)

        return F.linear(input, self.weight, self.bias)

    def get_weights(self, prefix):
        self.prefix = prefix
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.weight"] = self.weight_format_cast(self.weight.data)
        if self.bias is not None:
            weight_dict[f"{prefix}.bias"] = self.weight_format_cast(self.bias.data)
        return weight_dict
