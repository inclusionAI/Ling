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
from atb_llm.common_op_builders.activation.swiglu_common_op_builder import SwiGLUCommonOpBuilder
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.activation.test_swish_common_op_builder import ActKey


@ddt
class TestSwiGLUCommonOpBuilder(unittest.TestCase):
    def test_is_match_type_not_match(self):
        SwiGLUCommonOpBuilder().build = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWISH,
            "is_pack": True,
            "up_weight_only": False
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertNotIsInstance(builder, SwiGLUCommonOpBuilder)

    def test_is_match_is_pack_match(self):
        SwiGLUCommonOpBuilder().build = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWIGLU,
            "is_pack": True,
            "up_weight_only": False
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwiGLUCommonOpBuilder)

    @data((True, False), (False, False))
    @unpack
    def test_is_match_is_pack_up_weight_only_match(self, is_pack, up_weight_only):
        SwiGLUCommonOpBuilder().build = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWIGLU,
            "is_pack": is_pack,
            "up_weight_only": up_weight_only
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwiGLUCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build_is_pack_true(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWIGLU,
            "is_pack": True,
            "up_weight_only": False
        }
        act_tensor_map = {
            ActKey.input: ActKey.input,
            ActKey.act_out: ActKey.act_out
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwiGLUCommonOpBuilder)
        graph = builder.build(graph, act_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type="Activation",
                op_param='{"activationType": "ACTIVATION_SWIGLU_FORWARD", "dim": -1}',
                op_name='swiglu'),
        ])
        graph.add_operation.assert_has_calls([
            call(mock_atb_operation(),
                 [ActKey.input],
                 [ActKey.act_out])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_is_pack_false(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        act_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.ACTIVATION,
            "activation_type": ActivationType.SWIGLU,
            "is_pack": False,
            "up_weight_only": False
        }
        act_tensor_map = {
            ActKey.input: ActKey.input,
            ActKey.other_input: ActKey.other_input,
            ActKey.act_out: ActKey.act_out
        }
        builder = CommonOpBuilderManager.get_builder(act_param)
        self.assertIsInstance(builder, SwiGLUCommonOpBuilder)
        graph = builder.build(graph, act_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type="Concat", op_param='{"concatDim": -1}', op_name='test_Concat'),
            call(op_type="Activation",
                op_param='{"activationType": "ACTIVATION_SWIGLU_FORWARD", "dim": -1}',
                op_name='test_Swiglu'),
        ])
        graph.add_operation.assert_has_calls([
            call(mock_atb_operation(),
                 [ActKey.input, ActKey.other_input],
                 ["test_intermidiate_gate_up"]),
            call(mock_atb_operation(),
                 ["test_intermidiate_gate_up"],
                 [ActKey.act_out])
        ])
