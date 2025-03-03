# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import random
import torch
import torch_npu


class BaseFuzz:
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
        num_weight = random.randint(10, 80)
        for _ in range(num_weight):
            weight_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
        self.model.set_weight(weight_tensors)
    
    def set_kv_cache(self):
        kcache_tensors = []
        vcache_tensors = []
        num_kv = random.randint(10, 50)
        for _ in range(num_kv):
            kcache_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
            vcache_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
        self.model.set_kv_cache(kcache_tensors, vcache_tensors)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
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
        self.model.execute(input_tensors, acl_params)