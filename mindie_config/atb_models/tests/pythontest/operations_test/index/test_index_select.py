# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestIndexSelectOperation(operation_test.OperationTest):
    def setUp(self):
        self.dim = 0
        self.op_type = "IndexSelect"
        self.op_param = {"dim": 0}
        self.op_name = "IndexSelectOperation"
        self.op_set = (self.op_type, self.op_param, self.op_name)
        self.shape_0 = random.randint(1, 100)
        self.shape_1 = random.randint(1, 100)
        self.shape_2 = random.randint(1, 100)
        self.shape_3 = random.randint(1, 100)


    def get_golden(self, in_tensors):
        x = in_tensors['in0']
        indices = in_tensors['in1']
        torch.index_select(x, self.dim, indices)
        return [torch.index_select(x, self.dim, indices)]

    def test_2d(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            in_tensor_0 = torch.rand(self.shape_0, self.shape_1, dtype=dtype).npu()

            for dim in range(-2, 2):
                self.dim = dim
                self.op_param.update({"dim": self.dim})

                n = in_tensor_0.shape[dim]
                random_indices = torch.arange(n)
                for i in range(1, n):
                    in_tensor_1 = random_indices[:i].to(dtype=torch.int32).npu()
                    out_tensor_shape = list(in_tensor_0.shape)
                    out_tensor_shape[dim] = i
                    out_tensor_0 = torch.zeros(out_tensor_shape, dtype=dtype).npu()

                    inputs = {'in0': in_tensor_0, 'in1': in_tensor_1}
                    outputs = {'out0': out_tensor_0}

                    self.run_compare(self.op_set, inputs, outputs)

    def test_3d(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            in_tensor_0 = torch.rand(self.shape_0, self.shape_1, self.shape_2, dtype=dtype).npu()

            for dim in range(-3, 3):
                self.dim = dim
                self.op_param.update({"dim": self.dim})

                n = in_tensor_0.shape[dim]
                random_indices = torch.arange(n)
                for i in range(1, n):
                    in_tensor_1 = random_indices[:i].to(dtype=torch.int64).npu()
                    out_tensor_shape = list(in_tensor_0.shape)
                    out_tensor_shape[dim] = i
                    out_tensor_0 = torch.zeros(out_tensor_shape, dtype=dtype).npu()

                    inputs = {'in0': in_tensor_0, 'in1': in_tensor_1}
                    outputs = {'out0': out_tensor_0}

                    self.run_compare(self.op_set, inputs, outputs)

if __name__ == '__main__':
    unittest.main()
