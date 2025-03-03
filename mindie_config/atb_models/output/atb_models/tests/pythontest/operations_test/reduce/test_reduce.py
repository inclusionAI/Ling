# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest
import logging

import torch
import torch_npu
import torch.nn.functional as F

from operations_test import operation_test


class TestReduceOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Reduce"
        self.axis = [1]
        self.reduce_type = "REDUCE_SUM"
        self.op_param = {"reduceType":self.reduceType, "axis": self.axis}
        self.op_name = "ReduceOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = torch.sum(in_tensor['in0'], dim=self.axis[0])
        return [golden_out_tensor]

    def test_2d(self):
        in_tensor_0 = torch.rand(self.row, self.col).half().npu()
        out_tensor_0 = torch.zeros(self.row).half().npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)

    def test_2d_bf16(self):
        in_tensor_0 = torch.rand(self.row, self.col).npu().bfloat16()
        out_tensor_0 = torch.zeros(self.row).npu().bfloat16()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}
        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
