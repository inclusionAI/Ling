# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import unittest
from unittest.mock import Mock, patch, call

from ddt import ddt, data, unpack

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.qkv_linear.no_pack_common_op_builder import NoPackCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.qkv_linear.test_gqa_pack_common_op_builder import \
    QkvKey


@ddt
class TestNoPackCommonOpBuilder(unittest.TestCase):
    def test_is_match_is_pack_not_match(self):
        NoPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": True,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_module
            },
            "k_linear_param": {
                QkvKey.op_name: "test_k",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_module
            },
            "v_linear_param": {
                QkvKey.op_name: "test_v",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertNotIsInstance(builder, NoPackCommonOpBuilder)

    def test_is_match_is_pack_match(self):
        NoPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": False,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_module
            },
            "k_linear_param": {
                QkvKey.op_name: "test_k",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_module
            },
            "v_linear_param": {
                QkvKey.op_name: "test_v",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertIsInstance(builder, NoPackCommonOpBuilder)

    @data((LinearTypeV2.W8A16, 'W8A16MatMul'), (LinearTypeV2.W4A16, 'W4A16MatMul'))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_aclnn(self, linear_desc, op_type, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_modules = []
        for i in range(3):
            linear_module = LinearUtils()
            linear_module.prefix = f"test_{i}_module_prefix"
            linear_module.linear_desc = linear_desc
            linear_modules.append(linear_module)
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": False,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_modules[0],
            },
            "k_linear_param": {
                QkvKey.op_name: "test_k",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_modules[1]
            },
            "v_linear_param": {
                QkvKey.op_name: "test_v",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                QkvKey.linear_module: linear_modules[2]
            },
        }
        qkv_linear_tensor_map = {
            QkvKey.input: QkvKey.input,
            QkvKey.input_k: QkvKey.input_k,
            QkvKey.input_v: QkvKey.input_v,
            QkvKey.q_out: QkvKey.q_out,
            QkvKey.k_out: QkvKey.k_out,
            QkvKey.v_out: QkvKey.v_out
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertIsInstance(builder, NoPackCommonOpBuilder)
        graph = builder.build(graph, qkv_linear_tensor_map)
        graph.add_operation.assert_has_calls([
            call(mock_atb_operation(),
                 ["input", f"{linear_modules[0].prefix}.weight", f"{linear_modules[0].prefix}.weight_scale",
                  f"{linear_modules[0].prefix}.weight_offset"],
                 ["q_out"]),
            call(mock_atb_operation(),
                 ["input_k", f"{linear_modules[1].prefix}.weight", f"{linear_modules[1].prefix}.weight_scale",
                  f"{linear_modules[1].prefix}.weight_offset"],
                 ["k_out"]),
            call(mock_atb_operation(),
                 ["input_v", f"{linear_modules[2].prefix}.weight", f"{linear_modules[2].prefix}.weight_scale",
                  f"{linear_modules[2].prefix}.weight_offset"],
                 ["v_out"]),
        ])
