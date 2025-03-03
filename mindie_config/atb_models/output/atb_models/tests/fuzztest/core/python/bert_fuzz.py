# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import torch

from base_fuzz import BaseFuzz


class BertFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def execute_fa(self):
        weight_tensors = []
        num_weight = 21
        for i in range(num_weight):
            match i:
                case 0:
                    weight_tensors.append(torch.rand(250002, 1024, dtype=torch.float16).npu())
                case 1:
                    weight_tensors.append(torch.rand(514, 1024, dtype=torch.float16).npu())
                case 2:
                    weight_tensors.append(torch.rand(1, 1024, dtype=torch.float16).npu())
                case 5|7|9|11:
                    weight_tensors.append(torch.rand(1024, 1024, dtype=torch.float16).npu())
                case 15:
                    weight_tensors.append(torch.rand(4096, 1024, dtype=torch.float16).npu())
                case 16:
                    weight_tensors.append(torch.rand(4096, dtype=torch.float16).npu())
                case 17|19:
                    weight_tensors.append(torch.rand(1024, 4096, dtype=torch.float16).npu())
                case _:
                    weight_tensors.append(torch.rand(1024, dtype=torch.float16).npu())
        self.model.set_weight(weight_tensors)

        input_tensors = []
        input_tensors.append(torch.randint(1, 10, (2, 512), dtype=torch.int32).npu())   # input_ids
        input_tensors.append(torch.randint(1, 10, (2, 512), dtype=torch.int32).npu())   # position_ids
        input_tensors.append(torch.randint(1, 10, (2, 512), dtype=torch.int32).npu())   # token_type_ids
        input_tensors.append(torch.zeros((2, 512, 512), dtype=torch.float16).npu())     # attention mask
        input_tensors.append(torch.rand(1, dtype=torch.float16).npu())                  # k_cache
        input_tensors.append(torch.rand(1, dtype=torch.float16).npu())                  # v_cache
        input_tensors.append(torch.randint(1, 10, (2,), dtype=torch.int32).npu())       # token_offset
        input_tensors.append(torch.randint(1, 10, (2,), dtype=torch.int32).npu())       # seq_len
        input_tensors.append(torch.randint(1, 2, (1,), dtype=torch.int32).npu())        # layer_id

        acl_params = json.dumps({"seqLen": [512, 512], "tokenOffset": [512, 512]})

        self.model.execute(input_tensors, acl_params)