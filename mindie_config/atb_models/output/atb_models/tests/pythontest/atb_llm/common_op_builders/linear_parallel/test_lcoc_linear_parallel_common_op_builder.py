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
from atb_llm.common_op_builders.linear_parallel.lcoc_linear_parallel_common_op_builder import \
    LCOCLinearParallelCommonOpBuilder
from atb_llm.common_op_builders.linear_parallel.all_reduce_linear_parallel_common_op_builder import \
    AllReduceLinearParallelCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.linear.test_fp_linear_strategy import LinearKey


@ddt
class TestLCOCLinearParallelCommonOpBuilder(unittest.TestCase):
    @data((LinearTypeV2.FLOAT16,), (LinearTypeV2.BFLOAT16,))
    @unpack
    def test_is_match_linear_desc_match(self, linear_type):
        LCOCLinearParallelCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = linear_type
        linear_param = {
            'op_name': 'test_linear',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_parallel_param = {
            "op_name": "test_linear_parallel",
            "common_op_builder_category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
            "enable_lcoc": True
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertIsInstance(builder, LCOCLinearParallelCommonOpBuilder)

    @data((LinearTypeV2.W8A8,), (LinearTypeV2.W8A8S,),
          (LinearTypeV2.W8A8SC,), (LinearTypeV2.W4A16,), (LinearTypeV2.W8A16,))
    @unpack
    def test_is_match_linear_desc_not_match(self, linear_type):
        LCOCLinearParallelCommonOpBuilder().build = Mock()
        linear_module = LinearUtils()
        linear_module.linear_desc = linear_type
        linear_param = {
            'op_name': 'test_linear',
            "common_op_builder_category": CommonOpBuilderType.LINEAR,
            "linear_module": linear_module
        }
        linear_parallel_param = {
            "op_name": "test_linear_parallel",
            "common_op_builder_category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
            "enable_lcoc": True
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertNotIsInstance(builder, LCOCLinearParallelCommonOpBuilder)

    def test_is_match_comm_backend_not_match(self):
        LCOCLinearParallelCommonOpBuilder().build = Mock()
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
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.HCCL),
            "linear_param": linear_param,
            "enable_lcoc": True
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertNotIsInstance(builder, LCOCLinearParallelCommonOpBuilder)

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
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=0, world_size=1, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
            "enable_lcoc": True
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertIsInstance(builder, AllReduceLinearParallelCommonOpBuilder)
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
    def test_build_with_tp(self, mock_atb_operation):
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
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=0, world_size=2, backend=CommunicationBackend.LCCL),
            "linear_param": linear_param,
            "enable_lcoc": True
        }
        linear_tensor_map = {
            LinearKey.input: LinearKey.input,
            LinearKey.linear_out: LinearKey.linear_out
        }
        builder = CommonOpBuilderManager.get_builder(linear_parallel_param)
        self.assertIsInstance(builder, LCOCLinearParallelCommonOpBuilder)
        graph = builder.build(graph, linear_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type="LinearParallel",
                 op_param=json.dumps(
                     {"rank": 0, "rankSize": 2, "backend": "lccl", "rankTableFile": "", "transWeight": True}),
                 op_name='test_linear_parallel_LinearParallel')
        ])
        graph.add_operation.assert_called_once()
        graph.add_operation.assert_has_calls([
            call(ANY,
                 [LinearKey.input, "test_linear_module_prefix.weight"], [LinearKey.linear_out])
        ])
