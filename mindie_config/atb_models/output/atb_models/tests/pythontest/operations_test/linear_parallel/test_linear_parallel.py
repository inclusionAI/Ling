# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest
import json
import logging

import torch
import torch_npu
import torch.multiprocessing as mp

from operations_test import operation_test


def linear_parallel_worker(rank, world_size, world_input):
    torch_npu.npu.set_device(rank)
    m, k, n, x, weight = world_input

    # setUp
    op_type = "LinearParallel"
    op_param = {
        "rank": rank,
        "rankSize" : world_size,
        "rankRoot" : 0,
    }
    op_name = "LinearParallelOperation"

    # input for rowparallel
    slice_start, slice_end = k // world_size * rank, k // world_size * (rank + 1)
    if rank == world_size - 1:
        slice_end = k    

    in_tensor_0 = x[:, slice_start: slice_end]
    in_tensor_1 = weight[:, slice_start: slice_end]
    out_tensor_0 = torch.zeros(m, n, dtype=torch.float16)
    inputs = {'in0': in_tensor_0.npu(), 'in1': in_tensor_1.npu()}
    outputs = {'out0': out_tensor_0.npu()}

    # forward
    operation = operation_test.atb._BaseOperation(op_type=op_type,
                                   op_param=json.dumps(op_param),
                                        op_name=op_name)
    operation.forward(inputs, outputs)
    
    # gloden
    golden_out_tensor = torch.matmul(x.npu(), weight.npu().T)
    out_tensor = list(outputs.values())[0]

    if not golden_compare(out_tensor.cpu(), golden_out_tensor.cpu()):
        raise ValueError("Output tensor does not match the golden tensor.")


def golden_compare(out_tensor, golden_out_tensor, rtol=0.0001, atol=0.0001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        logger = logging.getLogger()
        logger.info("out_tensor.shape: %s, golden_out_tensor.shape: %s", out_tensor.shape, golden_out_tensor.shape)
        logger.info("out_tensor:\n %s", out_tensor)
        logger.info("golden_out_tensor:\n %s", golden_out_tensor)
    return result


class TestLinearParallelOperation(operation_test.OperationTest):
    def setUp(self):
        self.world_size = 2        
        self.m = random.randint(1, 100)
        self.k = random.randint(1, 100)
        self.n = random.randint(1, 100)

    def test_float16(self):
        world_input_x = torch.rand(self.m, self.k, dtype=torch.float16)
        world_input_weight = torch.rand(self.n, self.k, dtype=torch.float16)
        world_input = (self.m, self.k, self.n, world_input_x, world_input_weight)

        mp.spawn(linear_parallel_worker, nprocs=self.world_size, args=(self.world_size, world_input))
    
    
if __name__ == '__main__':
    unittest.main()