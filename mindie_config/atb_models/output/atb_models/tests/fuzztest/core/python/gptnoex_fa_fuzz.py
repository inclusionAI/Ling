# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import torch

from base_fuzz import BaseFuzz


class GPTNoexFaFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def execute_fa(self, acl_params, speculate_enable=None):
        weight_tensors = []
        num_weight = 15
        for i in range(num_weight):
            match i:
                case 0|15:
                    weight_tensors.append(torch.rand(50432, 6144).to(torch.half).npu())
                case 5:
                    weight_tensors.append(torch.rand(18432, 6144).to(torch.half).npu())
                case 6:
                    weight_tensors.append(torch.rand(18432).to(torch.half).npu())
                case 7:
                    weight_tensors.append(torch.rand(6144, 6144).to(torch.half).npu())
                case 9:
                    weight_tensors.append(torch.rand(24576, 6144).to(torch.half).npu())
                case 10:
                    weight_tensors.append(torch.rand(24576).to(torch.half).npu())
                case 11:
                    weight_tensors.append(torch.rand(6144, 24576).to(torch.half).npu())
                case _:
                    weight_tensors.append(torch.rand(6144).to(torch.half).npu())
        self.model.set_weight(weight_tensors)

        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (1, 13)).npu()) # input_ids
        input_tensors.append(torch.randint(1, 100, (1, 13)).npu()) # position_ids
        input_tensors.append(torch.randint(1, 100, (2048, 24)).npu()) # cos_embed
        input_tensors.append(torch.randint(1, 100, (2048, 24)).npu())  # sin_enbed
        input_tensors.append(torch.randint(1, 100, (1, 2048, 2048)).npu()) # attention mask
        input_tensors.append(torch.randint(1, 100, (1, 1, 2048, 6144)).npu()) # block_tables
        input_tensors.append(torch.randint(1, 100, (1, 1, 2048, 6144)).npu()) # slots
        input_tensors.append(torch.rand(1).npu())  # input_length
        input_tensors.append(torch.rand(1).npu())  # lm_head_indices
        input_tensors.append(torch.rand(1).npu())
        input_tensors.append(torch.rand(1).npu())
        input_tensors.append(torch.rand(1).npu())
        if speculate_enable:
            input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu())
        self.model.execute(input_tensors, acl_params)