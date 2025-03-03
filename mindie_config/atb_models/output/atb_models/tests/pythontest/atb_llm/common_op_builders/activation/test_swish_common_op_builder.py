# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import unittest
from unittest.mock import Mock, patch, call

from ddt import ddt, data, unpack

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, ActivationType
from atb_llm.common_op_builders.activation.swish_common_op_builder import SwishCommonOpBuilder
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph


class ActKey:
    input = 'input'
    other_input = 'other_input'
    act_out = 'act_out'


@ddt
class TestSwishCommonOpBuilder(unittest.TestCase):
    def test_is_match_type_not_match(self):
        SwishCommonOpBuilder().build = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWIGLU,
            "is_pack": True,
            "up_weight_only": True
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertNotIsInstance(builder, SwishCommonOpBuilder)

    def test_is_match_is_pack_match(self):
        SwishCommonOpBuilder().build = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWISH,
            "is_pack": True,
            "up_weight_only": True
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwishCommonOpBuilder)

    @data((True, True), (True, False), (False, False))
    @unpack
    def test_is_match_is_pack_up_weight_only_match(self, is_pack, up_weight_only):
        SwishCommonOpBuilder().build = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWISH,
            "is_pack": is_pack,
            "up_weight_only": up_weight_only
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwishCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build_is_pack_up_weight_only(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWISH,
            "is_pack": True,
            "up_weight_only": True
        }
        act_tensor_map = {
            ActKey.input: ActKey.input,
            ActKey.act_out: ActKey.act_out
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwishCommonOpBuilder)
        graph = builder.build(graph, act_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type="Split", op_param='{"splitDim": -1, "splitNum": 2}', op_name='test_Split'),
            call(op_type="Activation",
                op_param='{"activationType": "ACTIVATION_SWISH", "dim": -1}',
                op_name='test_Swish'),
        ])
        graph.add_operation.assert_has_calls([
            call(mock_atb_operation(),
                 [ActKey.input],
                 ["intermediate_gate", "intermediate_up"]),
            call(mock_atb_operation(),
                 ["intermediate_gate"],
                 [ActKey.act_out]),
        ])
