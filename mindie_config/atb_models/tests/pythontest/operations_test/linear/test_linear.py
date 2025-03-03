# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestLinearOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Linear"
        self.transpose_a = False
        self.transpose_b = False
        self.hasbias = False
        self.op_param = {
            "transposeA": self.transpose_a,
            "transposeB": self.transpose_b,
            "hasBias": self.hasbias
        }
        self.op_name = "LinearOperation"
        self.row = random.randint(1, 1000)
        self.col = random.randint(1, 1000)
        self.bias = 1e-5
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = torch.add(torch.matmul(in_tensor['in0'], in_tensor['in1']), self.bias).npu()
        return [golden_out_tensor]

    def test_float16(self):
        in_tensor_0 = torch.rand(10, 3, dtype=torch.float16).npu()
        in_tensor_1 = torch.rand(3, 3, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(10, 3, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0, 'in1': in_tensor_1.t()}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
