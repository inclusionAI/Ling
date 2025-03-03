# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import unittest
from unittest.mock import Mock, patch, call, ANY

import torch
from ddt import ddt, data, unpack

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.linear.atb_quant_linear_common_op_builder import ATBQuantLinearCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.linear.test_fp_linear_strategy import LinearKey


@ddt
class TestATBQuantLinearCommonOpBuilder(unittest.TestCase):
    def test_is_match_linear_type_not_match(self):
        ATBQuantLinearCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.W8A8SC
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module,
            "default_dtype": torch.bfloat16
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertNotIsInstance(builder, ATBQuantLinearCommonOpBuilder)

    @data((LinearTypeV2.W8A8,), (LinearTypeV2.W8A8S,), (LinearTypeV2.W8A8SC,))
    @unpack
    def test_is_match_linear_type_match(self, linear_type):
        ATBQuantLinearCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = linear_type
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ATBQuantLinearCommonOpBuilder)

    @data((torch.float16, "ACL_FLOAT16"), (torch.bfloat16, "ACL_BF16"))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_w8a8(self, torch_dtype, acl_dtype, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.W8A8
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module,
            "enable_quant_input": False,
            "default_dtype": torch_dtype
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ATBQuantLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Linear', op_param=f'{{"transposeB": true, "outDataType": "{acl_dtype}", "hasBias": true}}',
                 op_name='test_LinearQuant')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight", "test_linear_module_prefix.quant_bias",
                  "test_linear_module_prefix.deq_scale"],
                 [LinearKey.linear_out])
        ])

    @data((torch.float16, "ACL_FLOAT16"), (torch.bfloat16, "ACL_BF16"))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_w8a8_enable_quant_input(self, torch_dtype, acl_dtype, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.W8A8
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module,
            "enable_quant_input": True,
            "default_dtype": torch_dtype
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ATBQuantLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Elewise', op_param='{"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}', op_name='test_Elewise'),
            call(op_type='Linear', op_param=f'{{"transposeB": true, "outDataType": "{acl_dtype}", "hasBias": true}}',
                 op_name='test_LinearQuant')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.input_scale", "test_linear_module_prefix.input_offset"],
                 ["test_intermediate_input"]),
            call(ANY,
                 ["test_intermediate_input", "test_linear_module_prefix.weight", "test_linear_module_prefix.quant_bias",
                  "test_linear_module_prefix.deq_scale"],
                 [LinearKey.linear_out])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_w8a8sc(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.W8A8SC
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module,
            "enable_quant_input": False,
            "default_dtype": torch.float16
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ATBQuantLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='LinearSparse', op_param='{"tilingK": 8, "tilingN": 8}', op_name='test_LinearSparse')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight", "test_linear_module_prefix.quant_bias",
                  "test_linear_module_prefix.deq_scale", "test_linear_module_prefix.index"],
                 [LinearKey.linear_out])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_w8a8sc_enable_quant_input(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.W8A8SC
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module,
            "enable_quant_input": True,
            "default_dtype": torch.float16
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, ATBQuantLinearCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Elewise', op_param='{"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}', op_name='test_Elewise'),
            call(op_type='LinearSparse', op_param='{"tilingK": 8, "tilingN": 8}', op_name='test_LinearSparse')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.input_scale", "test_linear_module_prefix.input_offset"],
                 ["test_intermediate_input"]),
            call(ANY,
                 ["test_intermediate_input", "test_linear_module_prefix.weight", "test_linear_module_prefix.quant_bias",
                  "test_linear_module_prefix.deq_scale", "test_linear_module_prefix.index"],
                 [LinearKey.linear_out])
        ])
