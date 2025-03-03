# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest
import logging

import torch
import torch_npu

from operations_test import operation_test


class TestSplitOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Split"
        self.split_dim = -1
        self.split_num = 2
        self.op_param = {"splitDim": self.split_dim, "splitNum": self.split_num}
        self.op_name = "SplitOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = torch.chunk(in_tensor['in0'],
                                        chunks=self.split_num, dim=self.split_dim)
        return golden_out_tensor

    def test_2d(self):
        in_tensor_0 = torch.rand(self.row, 2 * self.col).half().npu()
        out_tensor_0 = torch.zeros(self.row, self.col).half().npu()
        out_tensor_1 = torch.zeros(self.row, self.col).half().npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0, 'out1': out_tensor_1}

        self.run_compare(self.op_set, inputs, outputs)

    def test_2d_bf16(self):
        in_tensor_0 = torch.rand(self.row, 2 * self.col).npu().bfloat16()
        out_tensor_0 = torch.zeros(self.row, self.col).npu().bfloat16()
        out_tensor_1 = torch.zeros(self.row, self.col).npu().bfloat16()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0, 'out1': out_tensor_1}
        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
