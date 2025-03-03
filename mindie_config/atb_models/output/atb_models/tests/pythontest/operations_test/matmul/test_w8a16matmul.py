# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestW8A16MatMulOperation(operation_test.OperationTest):
    def setUp(self):       
        self.op_type = "W8A16MatMul"
        self.has_bias = False
        self.quant_group_size = 0
        self.transposeb = False
        self.nd = True
        self.op_param = {
            "hasBias": self.has_bias,
            "quantGroupSize": self.quant_group_size,
            "transposeB": self.transposeb,
            "nd": self.nd
        }
        self.op_name = "W8A16MatMulOperation"
        self.m = random.randint(2, 100)
        self.n = random.randint(2, 200)
        self.k = random.randint(2, 300)
        self.op_set = (self.op_type, self.op_param, self.op_name)
            
    def get_golden(self, in_tensor):

        input_tensor, weight, antiquant_scale, antiquant_offset = \
            in_tensor['in0'], in_tensor['in1'], in_tensor['in2'], in_tensor['in3']

        golden_out_tensor = torch.mm(input_tensor, ((weight + antiquant_offset) * antiquant_scale))

        return [golden_out_tensor.npu()]

    
    def test_float16(self):
        input_tensor = torch.randint(-5, 5, (self.m, self.k), dtype=torch.float16).npu()
        weight = torch.randint(-5, 5, (self.k, self.n), dtype=torch.int8).npu()
        antiquant_scale = torch.randint(-5, 5, (1, self.n), dtype=torch.float16).npu()
        antiquant_offset = torch.randint(-5, 5, (1, self.n), dtype=torch.float16).npu()

        out_tensor_0 = torch.zeros(self.m, self.n, dtype=torch.float16).npu()
        
        inputs = {'in0': input_tensor, 'in1': weight, 'in2': antiquant_scale, 'in3': antiquant_offset}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)
  
    
if __name__ == '__main__':
    unittest.main() 
       