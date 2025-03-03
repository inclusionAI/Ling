# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import torch

from base_fuzz import BaseFuzz


class BaiChuanFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # input_ids
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # position_ids
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # cos_embed
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # sin_enbed
        input_tensors.append(torch.rand(random.randint(10, 200), random.randint(10, 200)).npu()) # attention mask
        input_tensors.append(torch.randint(1, 1024, (1, random.randint(10, 200))).npu()) # block_tables
        input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu()) # slots
        input_tensors.append(torch.rand(1).npu()) # placeholder
        input_tensors.append(torch.rand(1).npu()) # placeholder
        input_tensors.append(torch.rand(1).npu()) # placeholder
        input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # input_length
        input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # lm_head_indices
        if speculate_enable:
            input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu())
        self.model.execute(input_tensors, acl_params)