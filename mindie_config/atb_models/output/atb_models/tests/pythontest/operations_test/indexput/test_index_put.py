# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestIndexPutOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Indexput"
        self.op_param = {}
        self.op_name = "Indexput0"
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensors):
        if (
            "in0" not in in_tensors
            or "in1" not in in_tensors
            or "in2" not in in_tensors
        ):
            raise KeyError
        x = in_tensors["in0"]
        indices = in_tensors["in1"]
        values = in_tensors["in2"]
        x[indices] = values
        return [x]

    def test(self):
        full_seq = 4
        hidden = 4
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            in_tensor_0 = torch.randn([full_seq, hidden], dtype=dtype).npu()
            in_tensor_1 = torch.LongTensor(list(range(full_seq // 2))).npu()
            in_tensor_2 = torch.randn([full_seq // 2, hidden], dtype=dtype).npu()

            inputs = {"in0": in_tensor_0, "in1": in_tensor_1, "in2": in_tensor_2}
            outputs = {"out0": in_tensor_0}

            self.run_compare(self.op_set, inputs, outputs)


if __name__ == "__main__":
    unittest.main()
