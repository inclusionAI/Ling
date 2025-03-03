# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestElewiseCosOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Elewise"
        self.elewise_type = 'ELEWISE_COS'
        self.op_param = {'elewiseType': self.elewise_type}
        self.op_name = "ElewiseOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensors):
        golden_out_tensor = torch.cos(in_tensors['in0'])
        return [golden_out_tensor]

    def test_elewisecos_float16(self):
        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(self.row, self.col, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
