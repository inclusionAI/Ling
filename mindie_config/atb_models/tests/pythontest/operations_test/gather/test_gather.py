# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestGatherOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Gather"
        self.axis = 0
        self.batch_dims = 0
        self.op_param = {"axis": self.axis, "batchDims": self.batch_dims}
        self.op_name = "GatherOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.gather_row = random.randint(1, 100)
        self.gather_col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        x, indices = in_tensor['in0'], in_tensor['in1']
        if self.axis == 0:
            golden_out_tensor = x[indices, :]
        elif self.axis == 1:
            golden_out_tensor = x[:, indices]
        return [golden_out_tensor]

    def test_float16(self):
        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.float16)
        in_tensor_1 = torch.randint(0, self.row, (self.gather_row, self.gather_col), dtype=torch.int64)

        out_tensor_0 = torch.zeros(self.gather_row, self.gather_col, self.col, dtype=torch.float16)

        inputs = {'in0': in_tensor_0.npu(), 'in1': in_tensor_1.npu()}
        outputs = {'out0': out_tensor_0.npu()}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
