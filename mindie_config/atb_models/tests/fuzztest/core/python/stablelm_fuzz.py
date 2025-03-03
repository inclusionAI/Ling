# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import random
import torch
import torch_npu


class StablelmFuzz:
    def __init__(self, model_name):
        torch.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)
        self.model = torch.classes.ModelTorch.ModelTorch(model_name)

    def set_param(self, params):
        ret = self.model.set_param(params)
        return ret

    def set_weight(self):
        weight_tensors = []
        num_weight = 54
        for i in range(num_weight):
            match i:
                case 0:
                    weight_tensors.append(torch.rand(100353, 2048).to(torch.float16).npu())
                case 1 | 2 | 29 | 30 | 51 | 52:
                    weight_tensors.append(torch.rand(2048).to(torch.float16).npu())
                case 5:
                    weight_tensors.append(torch.rand(6144, 2048).to(torch.float16).npu())
                case 6:
                    weight_tensors.append(torch.rand(6144).to(torch.float16).npu())
                case 23:
                    weight_tensors.append(torch.rand(2048, 2048).to(torch.float16).npu())
                case 33:
                    weight_tensors.append(torch.rand(11264, 2048).to(torch.float16).npu())
                case 33:
                    weight_tensors.append(torch.rand(11264, 2048).to(torch.float16).npu())
                case 45:
                    weight_tensors.append(torch.rand(2048, 5632).to(torch.float16).npu())
                case 53:
                    weight_tensors.append(torch.rand(100352, 2048).to(torch.float16).npu())
                case _:
                    weight_tensors.append(torch.zeros(1).to(torch.float16).npu())
        self.model.set_weight(weight_tensors)

    def set_kv_cache(self):
        kcache_tensors = []
        vcache_tensors = []
        num_kv = 1
        for _ in range(num_kv):
            kcache_tensors.append(torch.rand(9, 128, 32, 64).to(torch.float16).npu())
            vcache_tensors.append(torch.rand(9, 128, 32, 64).to(torch.float16).npu())
        self.model.set_kv_cache(kcache_tensors, vcache_tensors)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (1024,)).to(torch.int32).npu())  # input_ids
        input_tensors.append(torch.randint(1, 100, (1024,)).to(torch.int32).npu())  # position_ids
        input_tensors.append(torch.rand(4096, 128).to(torch.float16).npu())  # cos_embed
        input_tensors.append(torch.rand(4096, 128).to(torch.float16).npu())  # sin_enbed
        input_tensors.append(torch.rand(128, 128).to(torch.float16).npu())  # attention mask
        input_tensors.append(torch.randint(1, 10, (1, 8)).to(torch.int32).npu())  # block_tables
        input_tensors.append(torch.randint(1, 10, (1024,)).to(torch.int32).npu())  # slots
        input_tensors.append(torch.rand(1).to(torch.float16).npu())  # placeholder
        input_tensors.append(torch.rand(1).to(torch.float16).npu())  # placeholder
        input_tensors.append(torch.rand(1).to(torch.float16).npu())  # placeholder
        input_tensors.append(torch.randint(1, 10, (1,)).to(torch.int32).npu())  # input_length
        input_tensors.append(torch.randint(1, 10, (1,)).to(torch.int32).npu())  # lm_head_indices
        self.model.execute(input_tensors, acl_params)