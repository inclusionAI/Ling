# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestRmsNormOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "RmsNorm"
        self.layer_type = 'RMS_NORM_NORM'
        self.op_param = {'layerType': self.layer_type}
        self.op_name = "RmsNormOperation"
        self.epsilon = 1e-5
        self.row = random.randint(1, 100)
        self.col = random.randint(2, 6)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensors):
        x = in_tensors['in0']
        gamma = in_tensors['in1']
        reducedims = []
        edim = x.dim() - gamma.dim()
        for i in range(gamma.dim()):
            reducedims.append(edim + i)
        rstd = torch.rsqrt(x.pow(2).mean(reducedims, keepdim=True) + self.epsilon)
        result = x * rstd
        golden_out_tensor = result * gamma
        return [golden_out_tensor]

    def test_rms_norm_norm(self):
        # 最后一维要和32字节对齐
        in_tensor_0 = torch.rand(1, 32 * self.col, dtype=torch.float16).npu()
        in_tensor_1 = torch.rand(1, 32 * self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(1, 32 * self.col, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0, 'in1': in_tensor_1}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
