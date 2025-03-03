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
from atb_llm.common_op_builders.gate_up.gate_up_pack_common_op_builder import GateUpPackCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph


class GateUpKey:
    op_name = 'op_name'
    category = 'common_op_builder_category'
    input = 'input'
    input_up = 'input_up'
    gate_up_out = 'gate_up_out'
    up_out = 'up_out'


@ddt
class TestGateUpPackCommonOpBuilder(unittest.TestCase):
    def test_is_match_is_pack_not_match(self):
        GateUpPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        gate_up_linear_param = {
            GateUpKey.op_name: 'test',
            GateUpKey.category: CommonOpBuilderType.GATE_UP,
            "is_pack": False,
            "linear_param": {
                "op_name": "test_q",
                GateUpKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        self.assertNotIsInstance(builder, GateUpPackCommonOpBuilder)

    def test_is_match_is_pack_match(self):
        GateUpPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        gate_up_linear_param = {
            GateUpKey.op_name: 'test',
            GateUpKey.category: CommonOpBuilderType.GATE_UP,
            "is_pack": True,
            "linear_param": {
                "op_name": "test_q",
                GateUpKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        self.assertIsInstance(builder, GateUpPackCommonOpBuilder)

    @data((LinearTypeV2.W8A16, 'W8A16MatMul'), (LinearTypeV2.W4A16, 'W4A16MatMul'))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_aclnn(self, linear_desc, op_type, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_module_prefix"
        linear_module.linear_desc = linear_desc
        gate_up_linear_param = {
            GateUpKey.op_name: 'test',
            GateUpKey.category: CommonOpBuilderType.GATE_UP,
            "is_pack": True,
            "linear_param": {
                "op_name": "test",
                GateUpKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module,
            },
        }
        gate_up_linear_tensor_map = {
            GateUpKey.input: GateUpKey.input,
            GateUpKey.gate_up_out: GateUpKey.gate_up_out,
        }
        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        self.assertIsInstance(builder, GateUpPackCommonOpBuilder)
        graph = builder.build(graph, gate_up_linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type=op_type, op_param='{"transposeB": true, "groupSize": 0, "hasBias": false}',
                 op_name=f'test_{op_type}')
        ])
        graph.add_operation.assert_has_calls([
            call(mock_atb_operation(),
                 [GateUpKey.input, f"{linear_module.prefix}.weight", f"{linear_module.prefix}.weight_scale",
                  f"{linear_module.prefix}.weight_offset"],
                 [GateUpKey.gate_up_out])
        ])
