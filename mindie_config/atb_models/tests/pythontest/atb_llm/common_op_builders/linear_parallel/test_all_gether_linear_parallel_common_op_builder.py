# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import json
import unittest
from unittest.mock import Mock, patch, call, ANY

from ddt import ddt, data, unpack

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.common_op_builders.linear_parallel.all_gather_linear_parallel_common_op_builder import \
    AllGatherLinearParallelCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.linear.test_fp_linear_strategy import LinearKey


@ddt
class TestAllGatherLinearParallelCommonOpBuilder(unittest.TestCase):
    def test_is_match_linear_parallel_type_match(self):
        AllGatherLinearParallelCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = LinearTypeV2.FLOAT16
        linear_param = {
            'op_name': 'test_linear',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_parallel_param = {
            "op_name": "test_linear_parallel",
            "common_op_builder_category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_GATHER,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertIsInstance(builder, AllGatherLinearParallelCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build_world_size_1(self, mock_atb_operation):
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
        linear_parallel_param = {
            "op_name": "test_linear_parallel",
            "common_op_builder_category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_GATHER,
            "parallel_info": TensorParallelInfo(rank=0, world_size=1, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertIsInstance(builder, AllGatherLinearParallelCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Linear', op_param='{"transposeB": true, "hasBias": false}', op_name='test_Linear')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY, [LinearKey.input, "test_linear_module_prefix.weight"], [LinearKey.linear_out])
        ])

    @data((True, [1, 0, 2]), (False, [1, 2, 0, 3]))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_with_tp(self, unpad_inputs, perm, mock_atb_operation):
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
        linear_parallel_param = {
            "op_name": "test_linear_parallel",
            "common_op_builder_category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_GATHER,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
            "unpad_inputs": unpad_inputs
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertIsInstance(builder, AllGatherLinearParallelCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Linear', op_param='{"transposeB": true, "hasBias": false}', op_name='test_Linear'),
            call(op_type="AllGather",
                 op_param=json.dumps({"rank": 0, "rankSize": 2, "backend": "lccl", "rankTableFile": ""}),
                 op_name='test_linear_parallel_AllGather'),
            call(op_type="Transpose", op_param=f'{{"perm": {perm}}}',
                 op_name='test_linear_parallel_Transpose')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight"],
                 ["test_linear_parallel_intermediate_linear_out"]),
            call(ANY,
                 ['test_linear_parallel_intermediate_linear_out'], ['test_linear_parallel_all_gather_out']),
            call(ANY,
                 ['test_linear_parallel_all_gather_out'], ['linear_out']),
        ])