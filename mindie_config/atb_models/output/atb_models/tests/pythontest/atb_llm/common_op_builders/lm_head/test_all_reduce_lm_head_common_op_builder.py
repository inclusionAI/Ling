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
from atb_llm.common_op_builders.lm_head.all_reduce_lm_head_common_op_builder import AllReduceLmHeadCommonOpBuilder
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.lm_head.test_all_gather_lm_head_common_op_builder import \
    LmHeadKey


@ddt
class TestAllReduceLmHeadCommonOpBuilder(unittest.TestCase):
    def test_is_match_linear_parallel_type_match(self):
        AllReduceLmHeadCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_param = {
            'op_name': 'test_linear',
            LmHeadKey.category: CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_parallel_param = {
            LmHeadKey.op_name: "test_linear_parallel",
            LmHeadKey.category: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
        }
        lm_head_param = {
            LmHeadKey.op_name: "test_lm_head",
            LmHeadKey.category: CommonOpBuilderType.LM_HEAD,
            "enable_linear_parallel": True,
            "linear_parallel_param": linear_parallel_param,
            "head_dim": 128
        }
        builder = CommonOpBuilderManager.get_builder(lm_head_param)
        self.assertIsInstance(builder, AllReduceLmHeadCommonOpBuilder)

    @data((True, 0, [0, 256], [-1, 128]), (False, 1, [0, 0, 256], [-1, -1, 128]))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_with_gather_ahead_true_tp(self, unpad_inputs, axis, offsets, size, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_param = {
            'op_name': 'test',
            LmHeadKey.category: CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_parallel_param = {
            LmHeadKey.op_name: "test_linear_parallel",
            LmHeadKey.category: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=2, world_size=4, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param
        }
        lm_head_param = {
            LmHeadKey.op_name: "test_lm_head",
            LmHeadKey.category: CommonOpBuilderType.LM_HEAD,
            "enable_linear_parallel": True,
            "linear_parallel_param": linear_parallel_param,
            "gather_ahead": True,
            "unpad_inputs": unpad_inputs,
            "head_dim": 128
        }
        tensor_map = {
            LmHeadKey.input: LmHeadKey.input,
            LmHeadKey.indices: LmHeadKey.indices,
            LmHeadKey.linear_out: LmHeadKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(lm_head_param)
        self.assertIsInstance(builder, AllReduceLmHeadCommonOpBuilder)
        graph = builder.build(graph, tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Gather', op_param=f'{{"axis": {axis}}}', op_name='test_lm_head_Gather'),
            call(op_type='Slice', op_param=f'{{"offsets": {offsets}, "size": {size}}}', op_name='test_lm_head_Slice'),
            call(op_type='Linear', op_param='{"transposeB": true, "hasBias": false}', op_name='test_Linear'),
            call(op_type="AllReduce",
                 op_param='{"rank": 2, "rankSize": 4, "backend": "lccl", "rankTableFile": ""}',
                 op_name='test_linear_parallel_AllReduce')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, [LmHeadKey.input, LmHeadKey.indices], ["test_lm_head_intermediate_gather_out"]),
            call(ANY, ["test_lm_head_intermediate_gather_out"], ["test_lm_head_intermediate_slice_out"]),
            call(ANY, ["test_lm_head_intermediate_slice_out", "test_linear_module_prefix.weight"],
                 [LmHeadKey.linear_out]),
            call(ANY,
                 ['linear_out'], ['linear_out']),
        ])

    @data((True, [0, 256], [-1, 128]), (False, [0, 0, 256], [-1, -1, 128]))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_with_gather_ahead_false_tp(self, unpad_inputs, offsets, size, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        linear_module = LinearUtils()
        linear_module.prefix = "test_linear_module_prefix"
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_param = {
            'op_name': 'test',
            LmHeadKey.category: CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_parallel_param = {
            LmHeadKey.op_name: "test_linear_parallel",
            LmHeadKey.category: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=2, world_size=4, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param
        }
        lm_head_param = {
            LmHeadKey.op_name: "test_lm_head",
            LmHeadKey.category: CommonOpBuilderType.LM_HEAD,
            "enable_linear_parallel": True,
            "linear_parallel_param": linear_parallel_param,
            "gather_ahead": False,
            "unpad_inputs": unpad_inputs,
            "head_dim": 128
        }
        tensor_map = {
            LmHeadKey.input: LmHeadKey.input,
            LmHeadKey.indices: LmHeadKey.indices,
            LmHeadKey.linear_out: LmHeadKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(lm_head_param)
        self.assertIsInstance(builder, AllReduceLmHeadCommonOpBuilder)
        graph = builder.build(graph, tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Slice', op_param=f'{{"offsets": {offsets}, "size": {size}}}', op_name='test_lm_head_Slice'),
            call(op_type='Linear', op_param='{"transposeB": true, "hasBias": false}', op_name='test_Linear'),
            call(op_type="AllReduce",
                 op_param='{"rank": 2, "rankSize": 4, "backend": "lccl", "rankTableFile": ""}',
                 op_name='test_linear_parallel_AllReduce')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, [LmHeadKey.input], ["test_lm_head_intermediate_slice_out"]),
            call(ANY, ["test_lm_head_intermediate_slice_out", "test_linear_module_prefix.weight"],
                 [LmHeadKey.linear_out]),
            call(ANY,
                 ['linear_out'], ['linear_out']),
        ])
