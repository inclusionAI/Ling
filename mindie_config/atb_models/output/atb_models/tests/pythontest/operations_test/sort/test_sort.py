# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest
import logging

import torch
import torch_npu

from operations_test import operation_test


class TestSortOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Sort"
        self.op_name = "SortOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.num = [random.randint(1, self.col)]
        self.op_param = {"num": self.num}
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = torch.topk(in_tensor['in0'], self.num[0])
        return [golden_out_tensor.values, golden_out_tensor.indices.to(torch.int32)]

    def test_2d(self):
        in_tensor_0 = torch.rand(self.row, self.col).half().npu()
        out_tensor_0 = torch.zeros(self.row, self.num[0]).half().npu()
        out_tensor_1 = torch.zeros(self.row, self.num[0], dtype=torch.int32).npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0, 'out1': out_tensor_1}

        self.run_compare(self.op_set, inputs, outputs)

    def test_2d_bf16(self):
        in_tensor_0 = torch.rand(self.row, self.col).npu().bfloat16()
        out_tensor_0 = torch.zeros(self.row, self.num[0]).npu().bfloat16()
        out_tensor_1 = torch.zeros(self.row, self.num[0], dtype=torch.int32).npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0, 'out1': out_tensor_1}
        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
