# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import torch
from base_fuzz import BaseFuzz


class InternLM2Fuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        if random.random() > 0.5:
            tensor_size = random.randint(10, 1024)
            input_tensors.append(torch.randint(1, 100, (tensor_size,)).npu()) # 0 input_ids
            input_tensors.append(torch.rand(1).npu()) # 1 placeholder
            input_tensors.append(torch.randint(1, 100, (tensor_size,)).npu()) # 2 position_ids
            input_tensors.append(torch.rand(32768, 128).npu()) # 3 cos_embed
            input_tensors.append(torch.rand(32768, 128).npu()) # 4 sin_enbed
            input_tensors.append(torch.rand(128, 128).npu()) # 5 attention mask
            input_tensors.append(torch.randint(1, 1024, (1, 8)).npu()) # 6 block_tables
            input_tensors.append(torch.randint(1, 1024, (tensor_size,)).npu()) # 7 slots
            input_tensors.append(torch.rand(1).npu()) # 8 placeholder
            input_tensors.append(torch.rand(1).npu()) # 9 placeholder
            input_tensors.append(torch.rand(1).npu()) # 10 placeholder
            input_tensors.append(torch.tensor([tensor_size]).npu()) # 11 input_length
            input_tensors.append(torch.tensor([tensor_size - 1]).npu()) # 12 lm_head_indices
            if speculate_enable:
                nput_tensors.append(torch.randint(1, 1024, (tensor_size,)).npu()) # 13 logits_indices
            input_tensors.append(torch.rand(1).npu()) # 14 im_mask
        else:
            input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # input_ids
            input_tensors.append(torch.rand(1).npu()) # placeholder
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
            input_tensors.append(torch.rand(1, 1024).npu() > 0.5) # im_mask
        self.model.execute(input_tensors, acl_params)
