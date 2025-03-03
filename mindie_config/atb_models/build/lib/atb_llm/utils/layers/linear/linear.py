# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class ColumnLinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 gather_output=True,
                 process_group=None):
        super().__init__()
        self.process_group = process_group
        self.tp_size = self.process_group.size()
        self.in_features = in_features
        self.out_features = out_features // self.tp_size
        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
        self.gather_output = gather_output
        self.bias = nn.Parameter(torch.ones(self.out_features)) if bias else None

    def multiply_gather(self,
                        x,
                        weight):
        x = x @ weight.t()

        if self.bias is not None:
            bias = self.bias.data.type(x.dtype)
            x = x + bias

        if self.gather_output and self.tp_size > 1:
            world_output = [
                torch.empty_like(x)
                for _ in range(self.process_group.size())
            ]
            torch.distributed.all_gather(world_output, x, group=self.process_group)
            x = torch.cat(world_output, dim=-1)

        return x

    def forward(self, x):
        return self.multiply_gather(x, self.weight.data)


class RowLinear(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 process_group=None):
        super().__init__()
        self.process_group = process_group
        self.tp_size = self.process_group.size()
        self.in_features = in_features // self.tp_size
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.ones(self.out_features)) if bias else None

    def multiply_reduce(self,
                        x,
                        weight):
        x = x @ weight.t()

        if self.tp_size > 1:
            torch.distributed.all_reduce(x, group=self.process_group)

        if self.bias is not None:
            bias = self.bias.data.type(x.dtype)
            x = x + bias

        return x

    def forward(self, x):
        return self.multiply_reduce(x, self.weight.data)