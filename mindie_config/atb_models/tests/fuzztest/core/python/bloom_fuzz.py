# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import torch

from base_fuzz import BaseFuzz


class BloomFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def set_weight(self):
        weight_tensors = []
        for _ in range(56):  # 56: bloom weight num
            weight_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
        self.model.set_weight(weight_tensors)
    
    def set_kv_cache(self):
        kcache_tensors = []
        vcache_tensors = []
        kcache_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
        vcache_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
        self.model.set_kv_cache(kcache_tensors, vcache_tensors)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # input_ids
        input_tensors.append(torch.rand(1).npu()) # placeholder
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
        self.model.execute(input_tensors, acl_params)
