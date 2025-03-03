# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import random
import torch
import torch_npu
from base_fuzz import BaseFuzz


class LlamaFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def _set_weight_w8(self):
        weight_tensors = []
        num_weight = 53
        for i in range(num_weight):
            match i:
                case 0|52:
                    weight_tensors.append(torch.rand(32000, 8192).to(torch.float16).npu())
                case 1|3|4|29|31|32|51:
                    weight_tensors.append(torch.rand(8192).to(torch.float16).npu())
                case 5:
                    weight_tensors.append(torch.randint(1, 10, (10240, 8192)).to(torch.int8).npu())
                case 8|9:
                    weight_tensors.append(torch.rand(10240, 1).to(torch.float16).npu())
                case 23:
                    weight_tensors.append(torch.randint(1, 10, (8192, 8192)).to(torch.int8).npu())
                case 26|27|48|49:
                    weight_tensors.append(torch.rand(8192, 1).to(torch.float16).npu())
                case 33:
                    weight_tensors.append(torch.randint(1, 10, (57344, 8192)).to(torch.int8).npu())
                case 36|37:
                    weight_tensors.append(torch.rand(57344, 1).to(torch.float16).npu())
                case 45:
                    weight_tensors.append(torch.randint(1, 10, (8192, 28672)).to(torch.int8).npu())
                case _:
                    weight_tensors.append(torch.zeros(1).to(torch.float16).npu())
        self.model.set_weight(weight_tensors)

    def _set_weight_w4(self):
        weight_tensors = []
        num_weight = 53
        for i in range(num_weight):
            match i:
                case 0|52:
                    weight_tensors.append(torch.rand(32000, 8192).to(torch.float16).npu())
                case 1|3|4|29|31|32|51:
                    weight_tensors.append(torch.rand(8192).to(torch.float16).npu())
                case 5:
                    weight_tensors.append(torch.randint(1, 10, (8192, 5120)).to(torch.int8).npu())
                case 8|9:
                    weight_tensors.append(torch.rand(64, 10240).to(torch.float16).npu())
                case 23:
                    weight_tensors.append(torch.randint(1, 10, (8192, 4096)).to(torch.int8).npu())
                case 26|27:
                    weight_tensors.append(torch.rand(64, 8192).to(torch.float16).npu())
                case 33:
                    weight_tensors.append(torch.randint(1, 10, (8192, 28672)).to(torch.int8).npu())
                case 36|37:
                    weight_tensors.append(torch.rand(64, 57344).to(torch.float16).npu())
                case 45:
                    weight_tensors.append(torch.randint(1, 10, (28672, 4096)).to(torch.int8).npu())
                case 48|49:
                    weight_tensors.append(torch.randint(1, 10, (224, 8192)).to(torch.float16).npu())
                case _:
                    weight_tensors.append(torch.zeros(1).to(torch.float16).npu())
        self.model.set_weight(weight_tensors)

    def _set_weight_w8a8(self):
        weight_tensors = []
        num_weight = 53
        for i in range(num_weight):
            match i:
                case 0|52:
                    weight_tensors.append(torch.rand(32000, 4096).to(torch.float16).npu())
                case 1|26|29|51:
                    weight_tensors.append(torch.rand(4096).to(torch.float16).npu())
                case 5:
                    weight_tensors.append(torch.randint(1, 10, (6144, 4096)).to(torch.int8).npu())
                case 8:
                    weight_tensors.append(torch.rand(6144).to(torch.float16).npu())
                case 9:
                    weight_tensors.append(torch.rand(6144).to(torch.float).npu())
                case 23:
                    weight_tensors.append(torch.randint(1, 10, (4096, 4096)).to(torch.int8).npu())
                case 27:
                    weight_tensors.append(torch.rand(4096).to(torch.float).npu())
                case 33:
                    weight_tensors.append(torch.randint(1, 10, (28672, 4096)).to(torch.int8).npu())
                case 36:
                    weight_tensors.append(torch.rand(28672).to(torch.float16).npu())
                case 37:
                    weight_tensors.append(torch.rand(28672).to(torch.float).npu())
                case 45:
                    weight_tensors.append(torch.rand(4096, 14336).to(torch.float16).npu())
                case _:
                    weight_tensors.append(torch.zeros(1).to(torch.float16).npu())
        self.model.set_weight(weight_tensors)

    def execute_quant(self, quant, acl_params, speculate_enable):
        if quant == 0:
            self._set_weight_w8() # w8a16
        elif quant == 1:
            self._set_weight_w4() # w4a16
        else:
            self._set_weight_w8a8() # w8a8

        kcache_tensors = []
        vcache_tensors = []
        num_kv = 1
        for i in range(num_kv):
            kcache_tensors.append(torch.rand(9, 128, 8, 128).to(torch.float16).npu())
            vcache_tensors.append(torch.rand(9, 128, 8, 128).to(torch.float16).npu())
        self.model.set_kv_cache(kcache_tensors, vcache_tensors)

        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (6,)).npu())
        input_tensors.append(torch.rand(1).to(torch.float16).npu())
        input_tensors.append(torch.randint(1, 100, (6,)).npu())
        if quant == 2:        
            input_tensors.append(torch.rand(131072, 128).to(torch.float16).npu())
            input_tensors.append(torch.rand(131072, 128).to(torch.float16).npu())
            input_tensors.append(torch.rand(1, 1).to(torch.float16).npu())
        else:
            input_tensors.append(torch.rand(8192, 128).to(torch.float16).npu())
            input_tensors.append(torch.rand(8192, 128).to(torch.float16).npu())
            input_tensors.append(torch.rand(128, 128).to(torch.float16).npu())
        input_tensors.append(torch.randint(1, 10, (1, 1)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 10, (6,)).to(torch.int32).npu())
        input_tensors.append(torch.rand(1).to(torch.float16).npu())
        input_tensors.append(torch.rand(1).to(torch.float16).npu())
        input_tensors.append(torch.rand(1).to(torch.float16).npu())
        input_tensors.append(torch.randint(1, 10, (1,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 10, (1,)).npu())
        if speculate_enable:
            input_tensors.append(torch.randint(1, 1024, (1024,)).npu())
        self.model.execute(input_tensors, acl_params)