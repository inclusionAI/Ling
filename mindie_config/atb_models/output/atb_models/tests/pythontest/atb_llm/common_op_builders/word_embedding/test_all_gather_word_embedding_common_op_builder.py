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
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import TensorParallelInfo, \
    CommunicationBackend
from atb_llm.common_op_builders.word_embedding.all_gather_word_embedding_common_op_builder import \
    AllGatherWordEmbeddingCommonOpBuilder
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph


class WordEmbKey:
    input_ids = 'input_ids'
    emb_weights = 'embedding_weights'
    word_emb_out = 'word_embedding_out'


@ddt
class TestAllGatherWordEmbeddingCommonOpBuilder(unittest.TestCase):
    def test_is_match(self):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        word_embedding_param = {
            "op_name": "test",
            "common_op_builder_category": CommonOpBuilderType.WORD_EMBEDDING,
            "enable_parallel": True,
            "parallel_info": TensorParallelInfo(rank=2, world_size=4, backend=CommunicationBackend.LCCL),
        }
        builder = CommonOpBuilderManager.get_builder(word_embedding_param)
        self.assertIsInstance(builder, AllGatherWordEmbeddingCommonOpBuilder)

    @data((True, 0, [1, 0, 2]), (False, 1, [1, 2, 0, 3]))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build(self, unpad_inputs, axis, perm, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        word_embedding_param = {
            "op_name": "test",
            "common_op_builder_category": CommonOpBuilderType.WORD_EMBEDDING,
            "enable_parallel": True,
            "unpad_inputs": unpad_inputs,
            "parallel_info": TensorParallelInfo(rank=2, world_size=4, backend=CommunicationBackend.LCCL),
        }
        word_embedding_tensor_map = {
            WordEmbKey.input_ids: WordEmbKey.input_ids,
            WordEmbKey.emb_weights: WordEmbKey.emb_weights,
            WordEmbKey.word_emb_out: WordEmbKey.word_emb_out
        }
        builder = CommonOpBuilderManager.get_builder(word_embedding_param)
        self.assertIsNotNone(builder)
        graph = builder.build(graph, word_embedding_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Gather', op_param=f'{{"axis": {axis}}}', op_name='test_Gather'),
            call(op_type="AllGather",
                 op_param='{"rank": 2, "rankSize": 4, "backend": "lccl", "rankTableFile": ""}',
                 op_name='test_AllGather'),
            call(op_type="Transpose", op_param=f'{{"perm": {perm}}}',
                 op_name='test_Transpose')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, [WordEmbKey.emb_weights, WordEmbKey.input_ids], ["test_gather_out"]),
            call(ANY, ["test_gather_out"], ["test_all_gather_out"]),
            call(ANY, ['test_all_gather_out'], ['word_embedding_out']),
        ])
