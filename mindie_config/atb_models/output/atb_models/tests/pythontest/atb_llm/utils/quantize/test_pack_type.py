# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import unittest
from unittest.mock import MagicMock

from ddt import ddt, data, unpack

from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import PackType, get_pack_type

FLOAT_UPPER = "FLOAT"
FLOAT_WEIGHT = "float"
NOT_EXIST_WEIGHT = "not_exist"
W8A8_WEIGHT = "w8a8"
W8A8S_WEIGHT = "w8a8s"
W8A8SC_WEIGHT = "w8a8sc"
W8A8_DYNAMIC_WEIGHT = "w8a8_dynamic"
W8A16_WEIGHT = "w8a16"
W4A16_WEIGHT = "w4a16"
NORM_WEIGHT = "norm"


@ddt
class TestPackType(unittest.TestCase):
    def setUp(self):
        self.weights = MagicMock()
        self.weights.quant_desc = {
            "float.weight": FLOAT_UPPER,
            "w4a16.weight": "W4A16",
            "w8a8.weight": "W8A8",
            "w8a16.weight": "W8A16",
            "w8a8s.weight": "W8A8S",
            "w8a8sc.weight": "W8A8SC",
            "w8a8_dynamic.weight": "W8A8_DYNAMIC",
            "norm.anti.weight": FLOAT_UPPER,
            "norm.module.weight": FLOAT_UPPER,
            "packed_w8a8sc.weight": "W8A8SC",
        }

    @data(
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_float(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.FLOAT
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8_WEIGHT, W8A8_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8),
        ([W8A8_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8),
        ([W8A8_WEIGHT, W8A8_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([W8A8_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8_WEIGHT], NOT_EXIST_WEIGHT, W8A8_WEIGHT, PackType.ALL_W8A8),
        ([W8A8_WEIGHT], NORM_WEIGHT, W8A8_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8S_WEIGHT, W8A8S_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8),
        ([W8A8S_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8),
        ([W8A8S_WEIGHT, W8A8S_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([W8A8S_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8S_WEIGHT], NOT_EXIST_WEIGHT, W8A8S_WEIGHT, PackType.ALL_W8A8),
        ([W8A8S_WEIGHT], NORM_WEIGHT, W8A8S_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8s(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8S
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8SC_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC),
        ([W8A8SC_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC_ANTI),
        ([W8A8SC_WEIGHT], NOT_EXIST_WEIGHT, W8A8SC_WEIGHT, PackType.ALL_W8A8SC),
        ([W8A8SC_WEIGHT], NORM_WEIGHT, W8A8SC_WEIGHT, PackType.ALL_W8A8SC_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8sc(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8SC
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8_DYNAMIC_WEIGHT, W8A8_DYNAMIC_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_DYNAMIC),
        ([W8A8_DYNAMIC_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_DYNAMIC),
        ([W8A8_DYNAMIC_WEIGHT, W8A8_DYNAMIC_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_DYNAMIC_ANTI),
        ([W8A8_DYNAMIC_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_DYNAMIC_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8_DYNAMIC_WEIGHT], NOT_EXIST_WEIGHT, W8A8_DYNAMIC_WEIGHT, PackType.ALL_W8A8_DYNAMIC),
        ([W8A8_DYNAMIC_WEIGHT], NORM_WEIGHT, W8A8_DYNAMIC_WEIGHT, PackType.ALL_W8A8_DYNAMIC_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8_dynamic(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8_DYNAMIC
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A16_WEIGHT, W8A16_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A16),
        ([W8A16_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A16),
        ([W8A16_WEIGHT, W8A16_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A16_ANTI),
        ([W8A16_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A16_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A16_WEIGHT], NOT_EXIST_WEIGHT, W8A16_WEIGHT, PackType.ALL_W8A16),
        ([W8A16_WEIGHT], NORM_WEIGHT, W8A16_WEIGHT, PackType.ALL_W8A16_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a16(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A16
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W4A16_WEIGHT, W4A16_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W4A16),
        ([W4A16_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W4A16),
        ([W4A16_WEIGHT, W4A16_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W4A16_ANTI),
        ([W4A16_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W4A16_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W4A16_WEIGHT], NOT_EXIST_WEIGHT, W4A16_WEIGHT, PackType.ALL_W4A16),
        ([W4A16_WEIGHT], NORM_WEIGHT, W4A16_WEIGHT, PackType.ALL_W4A16_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w4a16(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W4A16
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)


if __name__ == "__main__":
    unittest.main()
