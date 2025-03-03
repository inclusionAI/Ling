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
from atb_llm.common_op_builders.qkv_linear.gqa_pack_common_op_builder import GqaPackCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph


class QkvKey:
    op_name = 'op_name'
    category = 'common_op_builder_category'
    linear_module = 'linear_module'
    input = 'input'
    input_k = 'input_k'
    input_v = 'input_v'
    q_out = 'q_out'
    k_out = 'k_out'
    v_out = 'v_out'
    test_qkv = 'test_intermediate_mixed_qkv'


@ddt
class TestGqaPackCommonOpBuilder(unittest.TestCase):
    def test_is_match_head_num_not_match(self):
        GqaPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": True,
            "head_num": 64,
            "kv_head_num": 64,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertNotIsInstance(builder, GqaPackCommonOpBuilder)

    def test_is_match_head_num_match(self):
        GqaPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": True,
            "head_dim": 128,
            "head_num": 64,
            "kv_head_num": 32,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertIsInstance(builder, GqaPackCommonOpBuilder)
    
    @data((True, 32, 32), (False, 64, 64), (False, 64, 32))
    @unpack
    def test_is_match_pack_head_num_not_match(self, is_pack, head_num, kv_head_num):
        GqaPackCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": is_pack,
            "head_dim": 128,
            "head_num": head_num,
            "kv_head_num": kv_head_num,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module
            },
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertNotIsInstance(builder, GqaPackCommonOpBuilder)

    @data((LinearTypeV2.W8A16, 'W8A16MatMul'), (LinearTypeV2.W4A16, 'W4A16MatMul'))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_aclnn(self, linear_desc, op_type, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_qkv_module_prefix"
        linear_module.linear_desc = linear_desc
        qkv_linear_param = {
            QkvKey.op_name: 'test',
            QkvKey.category: CommonOpBuilderType.QKV,
            "is_pack": True,
            "head_dim": 128,
            "head_num": 64,
            "kv_head_num": 32,
            "linear_param": {
                QkvKey.op_name: "test_q",
                QkvKey.category: CommonOpBuilderType.LINEAR,
                "linear_module": linear_module,
                "enable_quant_input": False,
            }
        }
        qkv_linear_tensor_map = {
            QkvKey.input: QkvKey.input, QkvKey.q_out: QkvKey.q_out, 
            QkvKey.k_out: QkvKey.k_out, QkvKey.v_out: QkvKey.v_out
        }
        builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        self.assertIsInstance(builder, GqaPackCommonOpBuilder)
        graph = builder.build(graph, qkv_linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type=op_type, op_param='{"transposeB": true, "groupSize": 0, "hasBias": false}',
                 op_name=f'test_q_{op_type}'),
            call(op_type='Slice', op_param=f'{{"offsets": [0, 0], "size": [-1, {64 * 128}]}}', op_name='test_Slice_Q'),
            call(op_type='Slice',
                op_param=f'{{"offsets": [0, {64 * 128}], "size": [-1, {32 * 128 * 2}]}}',
                op_name='test_Slice_KV'),
            call(op_type='Split', op_param='{"splitDim": -1, "splitNum": 2}', op_name='test_Split'),
        ])
        graph.add_operation.assert_has_calls([
            call(mock_atb_operation(),
                 [QkvKey.input, f"{linear_module.prefix}.weight", f"{linear_module.prefix}.weight_scale",
                  f"{linear_module.prefix}.weight_offset"],
                 [QkvKey.test_qkv]),
            call(mock_atb_operation(),
                 [QkvKey.test_qkv],
                 [QkvKey.q_out]),
            call(mock_atb_operation(),
                 [QkvKey.test_qkv],
                 ["test_intermediate_mixed_kv"]),
            call(mock_atb_operation(),
                 ["test_intermediate_mixed_kv"],
                 [QkvKey.k_out, QkvKey.v_out])
        ])
