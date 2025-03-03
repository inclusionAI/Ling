# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import torch

from base_fuzz import BaseFuzz


class InternLMFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        if random.random() > 0.5:
            tensor_size = random.randint(10, 1024)
            input_tensors.append(torch.randint(1, 100, (tensor_size,)).npu()) # 0 input_ids
            input_tensors.append(torch.randint(1, 100, (tensor_size,)).npu()) # 1 position_ids
            input_tensors.append(torch.rand(32768, 128).npu()) # 2 cos_embed
            input_tensors.append(torch.rand(32768, 128).npu()) # 3 sin_enbed
            input_tensors.append(torch.rand(128, 128).npu()) # 4 attention mask
            input_tensors.append(torch.randint(1, 1024, (1, 8)).npu()) # 5 block_tables
            input_tensors.append(torch.randint(1, 1024, (tensor_size,)).npu()) # 6 slots
            input_tensors.append(torch.rand(1).npu()) # 7 placeholder
            input_tensors.append(torch.rand(1).npu()) # 8 placeholder
            input_tensors.append(torch.rand(1).npu()) # 9 placeholder
            input_tensors.append(torch.tensor([tensor_size]).npu()) # 10 sql_len
            input_tensors.append(torch.tensor([tensor_size - 1]).npu()) # 11 logits_indices
            if speculate_enable:
                input_tensors.append(torch.randint(1, 1024, (tensor_size,)).npu()) # 12 logits_indices
        else:
            input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # 0 input_ids
            input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # 1 position_ids
            input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # 2 cos_embed
            input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # 3 sin_enbed
            input_tensors.append(torch.rand(random.randint(10, 200), random.randint(10, 200)).npu()) # 4 attention mask
            input_tensors.append(torch.randint(1, 1024, (1, random.randint(10, 200))).npu()) # 5 block_tables
            input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu()) # 6 slots
            input_tensors.append(torch.rand(1).npu()) # 7 placeholder
            input_tensors.append(torch.rand(1).npu()) # 8 placeholder
            input_tensors.append(torch.rand(1).npu()) # 9 placeholder
            input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # 10 sql_len
            input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # 11 logits_indices
            if speculate_enable:
                input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu()) # 12
        self.model.execute(input_tensors, acl_params)
