# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import unittest
from unittest.mock import Mock, patch, call, ANY
from ddt import ddt, data, unpack

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.linear.fp_linear_common_op_builder import FpLinearCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph


class LinearKey:
    input = 'input'
    linear_out = 'linear_out'


@ddt
class TestFpLinearCommonOpBuilder(unittest.TestCase):
    @data((LinearTypeV2.FLOAT16,), (LinearTypeV2.BFLOAT16,))
    @unpack
    def test_is_match_linear_type_match(self, linear_type):
        FpLinearCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = linear_type
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, FpLinearCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build_without_bias(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, FpLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Linear', op_param='{"transposeB": true, "hasBias": false}', op_name='test_Linear')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight"], [LinearKey.linear_out])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_with_bias(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.has_bias = True
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            "bias": "bias",
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, FpLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Linear', op_param='{"transposeB": true, "hasBias": true}', op_name='test_Linear')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight", "test_linear_module_prefix.bias"],
                 [LinearKey.linear_out])
        ])

    @data((TransposeType.TRANSPOSE, '{"transposeB": true, "hasBias": false}'),
          (TransposeType.NOT_TRANSPOSE, '{"transposeB": false, "hasBias": false}'))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_with_transpose_b(self, transpose_b, op_param, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_module.trans_flag = transpose_b
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, FpLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Linear', op_param=op_param, op_name='test_Linear')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight"], [LinearKey.linear_out])
        ])
