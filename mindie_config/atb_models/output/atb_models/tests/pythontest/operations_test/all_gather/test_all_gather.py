# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest
import json
import logging

import torch
import torch_npu
import torch.multiprocessing as mp

from operations_test import operation_test


def all_gather_worker(rank, world_size, world_input):
    torch_npu.npu.set_device(rank)
    _, m, n = world_input.shape

    # setUp
    op_type = "AllGather"
    op_param = {
        "rank": rank,
        "rankSize": world_size,
        "rankRoot": 0,
    }
    op_name = "AllGatherOperation"

    # input
    in_tensor_0 = world_input[rank].clone().npu()
    out_tensor_0 = torch.zeros(world_size, m, n, dtype=torch.float16).npu()
    inputs = {'in0': in_tensor_0}
    outputs = {'out0': out_tensor_0}

    # forward
    operation = operation_test.atb._BaseOperation(op_type=op_type,
                                                  op_param=json.dumps(op_param),
                                                  op_name=op_name)
    operation.forward(inputs, outputs)

    # gloden
    golden_out_tensor = world_input
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


class TestALLGatherOperation(operation_test.OperationTest):
    def setUp(self):
        self.world_size = 2
        self.m = random.randint(1, 10)
        self.n = random.randint(1, 10)

    def test_float16(self):
        world_input = torch.rand(self.world_size, self.m, self.n, dtype=torch.float16)
        mp.spawn(all_gather_worker, nprocs=self.world_size, args=(self.world_size, world_input))


if __name__ == '__main__':
    unittest.main()
