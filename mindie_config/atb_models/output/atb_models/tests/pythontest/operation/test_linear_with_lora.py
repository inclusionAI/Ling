# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import unittest
import random
import json

import torch
import torch_npu
from torch.nn.utils.rnn import pad_sequence

from ddt import ddt, data


@ddt
class TestLinearWithLora(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_npu.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError(
                "env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path,
                                "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)

    def setUp(self):
        self.linear_with_lora_op = torch.classes.OperationTorch.OperationTorch("LinearWithLoraOperationCreate")
        self.m = random.randint(2, 10)
        self.n = random.randint(2, 20)
        self.k = random.randint(2, 30)
        self.r = 2 ** random.randint(1, 6)

    @data(torch.float16, torch.bfloat16)
    def test_all_zero_lora_tensor_no_gmm_no_bias(self, dtype):
        self.linear_with_lora_op.set_param(json.dumps({
            "isBF16": dtype == torch.bfloat16,
            "hasBias": False,
            "transposeType": True,
            "supportLora": True,
            "loraEnableGMM": False
        }))
        activation = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.n, self.k), dtype=dtype)
        placeholder = torch.zeros(1, dtype=dtype)
        lora_a = torch.zeros([self.k, self.r], dtype=dtype)
        lora_b = torch.zeros([self.r, self.n], dtype=dtype)
        operation_input = [activation.npu(), weight.npu()]
        operation_input.extend([placeholder.npu()] * 6)
        operation_input.extend([lora_a.T.contiguous().npu(), lora_b.npu()])
        npu_out = self.linear_with_lora_op.execute(operation_input)[0]

        # golden
        golden_out = torch.mm(activation.to(torch.float32), weight.T.to(torch.float32)).to(dtype)

        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))

    @data(torch.float16, torch.bfloat16)
    def test_no_gmm_no_bias(self, dtype):
        self.linear_with_lora_op.set_param(json.dumps({
            "isBF16": dtype == torch.bfloat16,
            "hasBias": False,
            "transposeType": True,
            "supportLora": True,
            "loraEnableGMM": False
        }))
        activation = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.n, self.k), dtype=dtype)
        placeholder = torch.zeros(1, dtype=dtype)
        lora_a = torch.randint(-5, 5, (self.k, self.r), dtype=dtype)
        lora_b = torch.randint(-5, 5, (self.r, self.n), dtype=dtype)
        operation_input = [activation.npu(), weight.npu()]
        operation_input.extend([placeholder.npu()] * 6)
        operation_input.extend([lora_a.T.contiguous().npu(), lora_b.npu()])
        npu_out = self.linear_with_lora_op.execute(operation_input)[0]

        # golden
        base_linear_out = torch.mm(activation.to(torch.float32), weight.T.to(torch.float32)).to(dtype)
        lora_a_linear_out = torch.mm(activation.to(torch.float32), lora_a.to(torch.float32))
        lora_b_linear_out = torch.mm(lora_a_linear_out, lora_b.to(torch.float32)).to(dtype)
        golden_out = base_linear_out + lora_b_linear_out

        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))

    @data(torch.float16, torch.bfloat16)
    def test_gmm_no_bias(self, dtype):
        self.linear_with_lora_op.set_param(json.dumps({
            "isBF16": dtype == torch.bfloat16,
            "hasBias": False,
            "transposeType": True,
            "supportLora": True,
            "loraEnableGMM": True
        }))
        input_1 = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        input_2 = torch.randint(-5, 5, (self.m * 2, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.n, self.k), dtype=dtype)
        placeholder = torch.zeros(1, dtype=dtype)
        lora_a_1 = torch.randint(-5, 5, (self.k, self.r), dtype=dtype)
        lora_a_2 = torch.randint(-5, 5, (self.k, self.r * 2), dtype=dtype)
        lora_b_1 = torch.randint(-5, 5, (self.r, self.n), dtype=dtype)
        lora_b_2 = torch.randint(-5, 5, (self.r * 2, self.n), dtype=dtype)
        activation = torch.cat((input_1, input_2), dim=0)
        lora_a = pad_sequence((lora_a_1.T, lora_a_2.T), batch_first=True)
        lora_b = pad_sequence((lora_b_1, lora_b_2), batch_first=True)
        operation_input = [activation.npu(), weight.npu()]
        operation_input.extend([placeholder.npu()] * 5)
        operation_input.extend(
            [torch.tensor([self.m, self.m * 3], dtype=torch.int64).npu(), lora_a.npu(), lora_b.npu()])
        npu_out = self.linear_with_lora_op.execute(operation_input)[0]

        # golden
        # adapter 1
        base_linear_out_1 = torch.mm(input_1.to(torch.float32), weight.T.to(torch.float32)).to(dtype)
        lora_a_linear_out_1 = torch.mm(input_1.to(torch.float32), lora_a_1.to(torch.float32))
        lora_b_linear_out_1 = torch.mm(lora_a_linear_out_1, lora_b_1.to(torch.float32)).to(dtype)
        golden_out_1 = base_linear_out_1 + lora_b_linear_out_1
        # adapter 2
        base_linear_out_2 = torch.mm(input_2.to(torch.float32), weight.T.to(torch.float32)).to(dtype)
        lora_a_linear_out_2 = torch.mm(input_2.to(torch.float32), lora_a_2.to(torch.float32))
        lora_b_linear_out_2 = torch.mm(lora_a_linear_out_2, lora_b_2.to(torch.float32)).to(dtype)
        golden_out_2 = base_linear_out_2 + lora_b_linear_out_2
        golden_out = torch.cat((golden_out_1, golden_out_2), dim=0)

        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))
