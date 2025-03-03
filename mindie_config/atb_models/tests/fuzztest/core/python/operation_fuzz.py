# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import random
import torch
import torch_npu


class OperationFuzz:
    def __init__(self, operation_name):
        torch.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)
        self.op = torch.classes.OperationTorch.OperationTorch(operation_name)

    def set_name(self, name):
        self.op.set_name(name)

    def set_param(self, params):
        self.op.set_param(params)

    def execute(self):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2

        self.op.execute(input_tensors)
    
    def execute_out(self):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2
        output_tensors = []
        output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # outtensor1
        output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # outtensor2

        self.op.execute_out(input_tensors, output_tensors)
    
    def execute_with_param(self, param):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2

        self.op.execute_with_param(input_tensors, param)
    
    def execute_out_with_param(self, param):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2
        output_tensors = []
        output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # outtensor1
        output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # outtensor2

        self.op.execute_out_with_param(input_tensors, output_tensors, param)