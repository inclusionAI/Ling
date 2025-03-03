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
from atb_llm.common_op_builders.word_embedding.no_parallel_word_embedding_common_op_builder import \
    NoParallelWordEmbeddingCommonOpBuilder
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from tests.pythontest.atb_llm.common_op_builders.word_embedding.test_all_gather_word_embedding_common_op_builder \
    import WordEmbKey


@ddt
class TestNoParallelWordEmbeddingCommonOpBuilder(unittest.TestCase):
    def test_is_match_enable_parallel_false(self):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        word_embedding_param = {
            "op_name": "test",
            "common_op_builder_category": CommonOpBuilderType.WORD_EMBEDDING,
            "enable_parallel": False,
            "parallel_info": TensorParallelInfo(rank=2, world_size=4, backend=CommunicationBackend.LCCL),
        }
        builder = CommonOpBuilderManager.get_builder(word_embedding_param)
        self.assertIsInstance(builder, NoParallelWordEmbeddingCommonOpBuilder)

    def test_is_match_enable_parallel_true(self):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        word_embedding_param = {
            "op_name": "test",
            "common_op_builder_category": CommonOpBuilderType.WORD_EMBEDDING,
            "enable_parallel": True,
            "parallel_info": TensorParallelInfo(rank=0, world_size=1, backend=CommunicationBackend.LCCL),
        }
        builder = CommonOpBuilderManager.get_builder(word_embedding_param)
        self.assertIsInstance(builder, NoParallelWordEmbeddingCommonOpBuilder)

    @data((True, 0), (False, 1))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build(self, unpad_inputs, axis, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        word_embedding_param = {
            "op_name": "test",
            "common_op_builder_category": CommonOpBuilderType.WORD_EMBEDDING,
            "enable_parallel": False,
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
        graph.add_operation.assert_called_once()
        mock_atb_operation.assert_has_calls([
            call(op_type='Gather', op_param=f'{{"axis": {axis}}}', op_name='test_Gather'),
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, [WordEmbKey.emb_weights, WordEmbKey.input_ids], [WordEmbKey.word_emb_out])
        ])
