# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import unittest
import random
import json

import torch
import torch_npu

from ddt import ddt, data


@ddt
class TestW8A16Matmul(unittest.TestCase):
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
        self.w8a16_matmul_op = torch.classes.OperationTorch.OperationTorch("W8A16Operation")
        self.m = random.randint(2, 10)
        self.n = random.randint(2, 20)
        self.k = random.randint(2, 30)

    @data(torch.float16, torch.bfloat16)
    def test_per_channel_nz_transpose_no_bias(self, dtype):
        self.w8a16_matmul_op.set_param(json.dumps({
            "hasBias": False,
            "quantGroupSize": 0,
            "transposeB": True,
            "nd": False
        }))
        input_tensor = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.n, self.k), dtype=torch.int8)
        weight_scale = torch.randint(-5, 5, (self.n, 1), dtype=dtype)
        weight_offset = torch.randint(-5, 5, (self.n, 1), dtype=dtype)

        # golden
        golden_out = torch.mm(
            input_tensor.to(torch.float32),
            ((weight.T - weight_offset.T) * weight_scale.T).to(torch.float32)
        ).to(dtype)

        # npu
        npu_weight = weight.npu()
        torch_npu.npu_format_cast_(npu_weight, 29)
        npu_out = self.w8a16_matmul_op.execute([
            input_tensor.npu(), npu_weight,
            weight_scale.npu(), -weight_offset.npu()
        ])[0]
        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))

    @data(torch.float16, torch.bfloat16)
    def test_per_channel_nz_no_transpose_no_bias(self, dtype):
        self.w8a16_matmul_op.set_param(json.dumps({
            "hasBias": False,
            "quantGroupSize": 0,
            "transposeB": False,
            "nd": False
        }))
        input_tensor = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.k, self.n), dtype=torch.int8)
        weight_scale = torch.randint(-5, 5, (1, self.n), dtype=dtype)
        weight_offset = torch.randint(-5, 5, (1, self.n), dtype=dtype)

        # golden
        golden_out = torch.mm(
            input_tensor.to(torch.float32),
            ((weight - weight_offset) * weight_scale).to(torch.float32)
        ).to(dtype)

        # npu
        npu_weight = weight.npu()
        torch_npu.npu_format_cast_(npu_weight, 29)
        npu_out = self.w8a16_matmul_op.execute([
            input_tensor.npu(), npu_weight,
            weight_scale.npu(), -weight_offset.npu()
        ])[0]
        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))

    @data(torch.float16, torch.bfloat16)
    def test_per_channel_nd_transpose_no_bias(self, dtype):
        self.w8a16_matmul_op.set_param(json.dumps({
            "hasBias": False,
            "quantGroupSize": 0,
            "transposeB": True,
            "nd": True
        }))
        input_tensor = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.n, self.k), dtype=torch.int8)
        weight_scale = torch.randint(-5, 5, (self.n, 1), dtype=dtype)
        weight_offset = torch.randint(-5, 5, (self.n, 1), dtype=dtype)

        # golden
        golden_out = torch.mm(
            input_tensor.to(torch.float32),
            ((weight.T - weight_offset.T) * weight_scale.T).to(torch.float32)
        ).to(dtype)

        # npu
        npu_weight = weight.npu()
        npu_out = self.w8a16_matmul_op.execute([
            input_tensor.npu(), npu_weight,
            weight_scale.npu(), -weight_offset.npu()
        ])[0]
        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))

    @data(torch.float16, torch.bfloat16)
    def test_per_channel_nd_no_transpose_no_bias(self, dtype):
        self.w8a16_matmul_op.set_param(json.dumps({
            "hasBias": False,
            "quantGroupSize": 0,
            "transposeB": False,
            "nd": False
        }))
        input_tensor = torch.randint(-5, 5, (self.m, self.k), dtype=dtype)
        weight = torch.randint(-5, 5, (self.k, self.n), dtype=torch.int8)
        weight_scale = torch.randint(-5, 5, (1, self.n), dtype=dtype)
        weight_offset = torch.randint(-5, 5, (1, self.n), dtype=dtype)

        # golden
        golden_out = torch.mm(
            input_tensor.to(torch.float32),
            ((weight - weight_offset) * weight_scale).to(torch.float32)
        ).to(dtype)

        # npu
        npu_weight = weight.npu()
        npu_out = self.w8a16_matmul_op.execute([
            input_tensor.npu(), npu_weight,
            weight_scale.npu(), -weight_offset.npu()
        ])[0]
        self.assertTrue(torch.equal(golden_out.cpu(), npu_out.cpu()))
