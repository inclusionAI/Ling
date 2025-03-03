# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest
import logging

import torch
import torch_npu

from operations_test import operation_test


class TestSoftmaxOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Softmax"
        self.axes = [1]
        self.op_param = {"axes": self.axes}
        self.op_name = "SoftmaxOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = torch.softmax(in_tensor['in0'], dim=self.axes[0])
        return [golden_out_tensor]

    def test_2d(self):
        in_tensor_0 = torch.rand(self.row, self.col).half().npu()
        out_tensor_0 = torch.zeros(self.row, self.col).half().npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)

    def test_2d_bf16(self):
        in_tensor_0 = torch.rand(self.row, self.col).npu().bfloat16()
        out_tensor_0 = torch.zeros(self.row, self.col).npu().bfloat16()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}
        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
