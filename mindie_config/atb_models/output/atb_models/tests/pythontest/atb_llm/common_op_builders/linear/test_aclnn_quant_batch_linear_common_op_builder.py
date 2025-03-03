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
from atb_llm.common_op_builders.linear.aclnn_quant_batch_linear_common_op_builder import \
    ACLNNQuantBatchLinearCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.linear.test_fp_linear_strategy import LinearKey


@ddt
class TestACLNNQuantBatchLinearCommonOpBuilder(unittest.TestCase):
    @data((LinearTypeV2.W4A16,), (LinearTypeV2.W8A16,))
    @unpack
    def test_is_match_linear_type_match(self, linear_type):
        ACLNNQuantBatchLinearCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = linear_type
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ACLNNQuantBatchLinearCommonOpBuilder)

    @data((LinearTypeV2.W8A16, 'W8A16MatMul'), (LinearTypeV2.W4A16, 'W4A16MatMul'))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_add_operation(self, linear_desc, op_type, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = linear_desc
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
        self.assertIsInstance(builder, ACLNNQuantBatchLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type=op_type, op_param='{"transposeB": true, "groupSize": 0, "hasBias": false}',
                 op_name=f'test_{op_type}')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight", "test_linear_module_prefix.weight_scale",
                  "test_linear_module_prefix.weight_offset"],
                 [LinearKey.linear_out])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_with_bias(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.W4A16
        linear_module.has_bias = True
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
        self.assertIsInstance(builder, ACLNNQuantBatchLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type="W4A16MatMul", op_param='{"transposeB": true, "groupSize": 0, "hasBias": true}',
                 op_name='test_W4A16MatMul')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight", "test_linear_module_prefix.weight_scale",
                  "test_linear_module_prefix.weight_offset", "test_linear_module_prefix.bias"],
                 [LinearKey.linear_out])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_with_group_size(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.W4A16
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module,
            "group_size": 128
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ACLNNQuantBatchLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type="W4A16MatMul", op_param='{"transposeB": true, "groupSize": 128, "hasBias": false}',
                 op_name='test_W4A16MatMul')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight", "test_linear_module_prefix.weight_scale",
                  "test_linear_module_prefix.weight_offset"],
                 [LinearKey.linear_out])
        ])
