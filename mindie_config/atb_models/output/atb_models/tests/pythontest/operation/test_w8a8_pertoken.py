# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import unittest
import torch
import torch_npu
from loguru import logger
from ddt import ddt, data, unpack

CPU_DEVICE = "cpu"


@ddt
class TestW8A8Pertoken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_npu.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)

    def setUp(self):
        self.w8a8_op = torch.classes.OperationTorch.OperationTorch("W8A8Operation")
        self.m = 2
        self.k = 3
        self.n = 4

    @data((torch.float16, False), (torch.bfloat16, False), (torch.float16, True), (torch.bfloat16, True))
    @unpack
    def test_w8a8_pertoken_symmetric_has_bias(self, dtype, transpose_b):
        self.w8a8_op.set_param(json.dumps({
            "hasBias": True,
            "transposeB": transpose_b
        }))

        input_tensor = torch.randint(low=-128, high=127, size=(self.m, self.k), dtype=torch.int8, device=CPU_DEVICE)
        weight = torch.randint(low=-128, high=127, size=(self.k, self.n), dtype=torch.int8, device=CPU_DEVICE)
        weight_scale = torch.randn((self.n), dtype=torch.float32, device=CPU_DEVICE)
        if dtype == torch.bfloat16:
            weight_scale = weight_scale.to(torch.bfloat16)
        in_tensor_scale = torch.randn((self.m), dtype=torch.float32, device=CPU_DEVICE)
        quant_bias = torch.ones((self.n), dtype=torch.int32, device=CPU_DEVICE)

        # golden
        golden_out = (torch.mm(input_tensor.to(torch.float32), weight.to(torch.float32)) + quant_bias) * weight_scale
        golden_out = (golden_out.T * in_tensor_scale).T
        golden_out = golden_out.to(dtype)
        logger.info("input: %s, %s, %s" % (input_tensor.shape, input_tensor.dtype, input_tensor))
        logger.info("weight: %s, %s, %s" % (weight.shape, weight.dtype, weight))
        logger.info("golden out: %s, %s, %s" % (golden_out.shape, golden_out.dtype, golden_out))

        if transpose_b:
            weight = weight.T.contiguous()

        # npu
        w8a8_output = self.w8a8_op.execute([
            input_tensor.contiguous().npu(),
            weight.contiguous().npu(),
            weight_scale.contiguous().npu(),
            in_tensor_scale.contiguous().npu(),
            quant_bias.contiguous().npu(),
        ])[0]

        logger.info("w8a8 output: %s, %s, %s" % (w8a8_output.shape, w8a8_output.dtype, w8a8_output))

        self.assertTrue(torch.equal(golden_out.cpu(), w8a8_output.cpu()))
    
    @data((torch.float16, True), (torch.bfloat16, True), (torch.float16, False), (torch.bfloat16, False))
    @unpack
    def test_w8a8_pertoken_symmetric_no_bias(self, dtype, transpose_b):
        self.w8a8_op.set_param(json.dumps({
            "hasBias": False,
            "transposeB": transpose_b,
            "isAsymmetric": False,
        }))

        input_tensor = torch.randint(low=-128, high=127, size=(self.m, self.k), dtype=torch.int8, device=CPU_DEVICE)
        weight = torch.randint(low=-128, high=127, size=(self.k, self.n), dtype=torch.int8, device=CPU_DEVICE)
        weight_scale = torch.randn((self.n), dtype=torch.float32, device=CPU_DEVICE)
        if dtype == torch.bfloat16:
            weight_scale = weight_scale.to(torch.bfloat16)
        input_tensor_scale = torch.randn((self.m), dtype=torch.float32, device=CPU_DEVICE)

        # golden
        golden_out = torch.mm(input_tensor.to(torch.float32), weight.to(torch.float32)) * weight_scale
        golden_out = (golden_out.T * input_tensor_scale).T
        golden_out = golden_out.to(dtype)
        logger.info("input: %s, %s, %s" % (input_tensor.shape, input_tensor.dtype, input_tensor))
        logger.info("weight: %s, %s, %s" % (weight.shape, weight.dtype, weight))
        logger.info("golden out: %s, %s, %s" % (golden_out.shape, golden_out.dtype, golden_out))

        if transpose_b:
            weight = weight.T.contiguous()

        # npu
        w8a8_output = self.w8a8_op.execute([
            input_tensor.npu(),
            weight.npu(),
            weight_scale.npu(),
            input_tensor_scale.npu(),
        ])[0]

        logger.info("w8a8 output: %s, %s, %s" % (w8a8_output.shape, w8a8_output.dtype, w8a8_output))

        self.assertTrue(torch.equal(golden_out.cpu(), w8a8_output.cpu()))


if __name__ == "__main__":
    unittest.main()
