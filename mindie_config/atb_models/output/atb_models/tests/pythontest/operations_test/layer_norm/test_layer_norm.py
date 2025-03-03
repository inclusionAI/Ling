# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu
import torch.nn.functional as F

from operations_test import operation_test


class TestLayerNormOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "LayerNorm"
        self.layer_type = "LAYER_NORM_NORM"
        self.quant_type = "QUANT_UNDEINFED"
        self.epsilon = 1e-5
        self.begin_norm_axis = random.randint(0, 1)
        self.begin_params_axis = 0
        self.op_param = {
            "layerType": self.layer_type,
            "normParam": {
                "quantType": self.quant_type,
                "epsilon ": self.epsilon,
                "beginNormAxis": self.begin_norm_axis,
                "beginParamsAxis": self.begin_params_axis,
            }
        }
        self.op_name = "LayerNormOperation"
        self.m = random.randint(2, 100)
        self.n = random.randint(2, 200)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):

        input_tensor, gamma, beta = \
            in_tensor['in0'], in_tensor['in1'], in_tensor['in2']
        input_tensor, gamma, beta = input_tensor.type(
            torch.float32), gamma.type(torch.float32), beta.type(torch.float32)

        normalized_shape = input_tensor.shape[self.begin_norm_axis:]
        new_gamma_shape = (1,) * self.begin_norm_axis + gamma.shape
        gamma = gamma.view(new_gamma_shape)
        beta = beta.view(new_gamma_shape)

        golden_out_tensor = F.layer_norm(
            input_tensor, normalized_shape, weight=gamma, bias=beta, eps=self.epsilon)

        return [golden_out_tensor.type(torch.float16).npu()]

    def test_float16(self):
        input_tensor = torch.rand(self.m, self.n, dtype=torch.float16).npu()
        param_shape = input_tensor.shape[self.begin_norm_axis:]
        gamma = torch.rand(size=param_shape, dtype=torch.float16).npu()
        beta = torch.rand(size=param_shape, dtype=torch.float16).npu()

        out_tensor_0 = torch.zeros(self.m, self.n, dtype=torch.float16).npu()

        inputs = {'in0': input_tensor, 'in1': gamma, 'in2': beta}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
