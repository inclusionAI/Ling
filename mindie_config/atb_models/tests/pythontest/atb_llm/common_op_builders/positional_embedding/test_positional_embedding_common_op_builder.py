# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import unittest
from unittest.mock import Mock, patch, call, ANY
from ddt import ddt

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.positional_embedding.positional_embedding_common_op_builder import \
    PositionalEmbeddingCommonOpBuilder
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph


@ddt
class TestPositionalEmbeddingCommonOpBuilder(unittest.TestCase):
    def test_is_match(self):
        PositionalEmbeddingCommonOpBuilder().build = Mock()
        linear_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.POSITIONAL_EMBEDDING,
        }
        builder = CommonOpBuilderManager.get_builder(linear_param)
        self.assertIsInstance(builder, PositionalEmbeddingCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        positional_embedding_param = {
            'op_name': 'test',
            "common_op_builder_category": CommonOpBuilderType.POSITIONAL_EMBEDDING,
        }
        pos_key = 'pos_ids'
        positional_embedding_tensor_map = {
            "cos_table": "cos_t",
            "sin_table": "sin_t",
            "position_ids": pos_key,
            "cos_embedding": "cos_emb",
            "sin_embedding": "sin_emb",
        }
        builder = CommonOpBuilderManager.get_builder(positional_embedding_param)
        self.assertIsInstance(builder, PositionalEmbeddingCommonOpBuilder)
        graph = builder.build(graph, positional_embedding_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='Gather', op_param='{}', op_name='test_Gather_cosine_table'),
            call(op_type='Gather', op_param='{}', op_name='test_Gather_sine_table')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, ["cos_t", pos_key], ["cos_emb"]),
            call(ANY, ["sin_t", pos_key], ["sin_emb"])
        ])
