# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)


class TestRopeOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Rope"
        self.rotary_coeff = 4
        self.op_param = {"rotaryCoeff": self.rotary_coeff}
        self.op_name = "RopeOperation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.ntoken = 4
        self.seqlen = 4
        self.hidden_size = 4096
        self.head_size = 128
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensors):
        seqlen = int(in_tensors['in4'][0])
        batch = self.ntoken // seqlen
        head_size = in_tensors['in2'].size()[1]
        head_num = self.hidden_size // head_size
        qlayer = in_tensors['in0'].view(seqlen, batch, head_num, head_size)
        q0, q1 = qlayer.chunk(2, -1)
        klayer = in_tensors['in1'].view(seqlen, batch, head_num, head_size)
        k0, k1 = klayer.chunk(2, -1)
        cos0, cos1 = in_tensors['in2'].unsqueeze(1).unsqueeze(1).chunk(2, -1)
        sin0, sin1 = in_tensors['in3'].unsqueeze(1).unsqueeze(1).chunk(2, -1)
        q0 = (q0 * cos0) + (rotate_half(q0) * sin0)
        k0 = (k0 * cos0) + (rotate_half(k0) * sin0)
        q1 = (q1 * cos1) + (rotate_half(q1) * sin1)
        k1 = (k1 * cos1) + (rotate_half(k1) * sin1)
        q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(self.ntoken, self.hidden_size)
        k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(self.ntoken, self.hidden_size)
        return [q, k]

    def test_float16(self):
        in_tensor_0 = torch.rand(self.ntoken, self.hidden_size, dtype=torch.float16).npu()
        in_tensor_1 = torch.rand(self.ntoken, self.hidden_size, dtype=torch.float16).npu()
        cos = torch.rand(self.ntoken, self.head_size, dtype=torch.float16).npu()
        sin = torch.rand(self.ntoken, self.head_size, dtype=torch.float16).npu()
        in_tensor_4 = torch.tensor([self.seqlen], dtype=torch.int32).npu()
        out_tensor_0 = torch.zeros(self.ntoken, self.hidden_size, dtype=torch.float16).npu()
        out_tensor_1 = torch.zeros(self.ntoken, self.hidden_size, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0, 'in1': in_tensor_1, 'in2': cos, 'in3': sin, 'in4': in_tensor_4}
        outputs = {'out0': out_tensor_0, 'out1': out_tensor_1}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
