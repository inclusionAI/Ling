# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import unittest
from unittest.mock import Mock, patch, call, ANY
from ddt import ddt, data, unpack

import torch

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import AttnType
from atb_llm.common_op_builders.attention.atb_decoder_paged_attention_common_op_builder import \
    ATBDecoderPagedAttentionCommonOpBuilder
from atb_llm.common_op_builders.data_type import OperationBackend
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.layers.attention.kv_cache import KvCache


class TensorName:
    k_cache = 'k_cache'
    v_cache = 'v_cache'
    slots = 'slots'
    block_tables = 'block_tables'
    seq_len = 'seq_len'
    ra_seq_len = 'ra_seq_len'
    q_len = 'q_len'
    attention_out = 'attention_out'
    attention_mask = 'attention_mask'
    batch_wins = 'batch_wins'
    layer_id = 'layer_id'
    token_offset = 'token_offset'
    slopes = 'slopes'
    test_reshape_q = 'test_attention_reshape_q'
    test_reshape_k = 'test_attention_reshape_k'
    test_reshape_v = 'test_attention_reshape_v'


@ddt
class TestATBDecoderPagedAttentionCommonOpBuilder(unittest.TestCase):
    @data((True,), (False,))
    @unpack
    def test_is_match(self, enable_kv_quant):
        ATBDecoderPagedAttentionCommonOpBuilder().build = Mock()
        kv_quant_module = KvCache(torch.rand([10]), torch.rand([10]), torch.float16)
        attention_param = {
            'op_name': 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": False,
            "enable_kv_quant": enable_kv_quant,
            "kv_quant_module": kv_quant_module,
            "atb_attention_param": {},
            "operation_backend": OperationBackend.ATB,
            "atb_reshape_and_cache_param": {}
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBDecoderPagedAttentionCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build_kv_quant_enable_false(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        graph.add_reshape = Mock()
        attention_param = {
            'op_name': 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": False,
            "enable_kv_quant": False,
            "atb_attention_param": {},
            "operation_backend": OperationBackend.ATB,
            "atb_reshape_and_cache_param": {}
        }
        attention_tensor_map = {
            "q": "q",
            "k": "k",
            "v": "v",
            TensorName.k_cache: TensorName.k_cache,
            TensorName.v_cache: TensorName.v_cache,
            TensorName.slots: TensorName.slots,
            TensorName.block_tables: TensorName.block_tables,
            TensorName.seq_len: TensorName.seq_len,
            TensorName.attention_out: TensorName.attention_out
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBDecoderPagedAttentionCommonOpBuilder)
        graph = builder.build(graph, attention_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='ReshapeAndCache', op_param='{}', op_name='test_attention_ReshapeAndCache'),
            call(op_type='PagedAttention', op_param='{}', op_name='test_attention_PagedAttention')
        ])
        graph.add_reshape.assert_has_calls([
            call("q", "test_attention_reshape_q", builder.reshape_q),
            call("k", "test_attention_reshape_k", builder.reshape_kv),
            call("v", "test_attention_reshape_v", builder.reshape_kv),
            call("test_attention_intermediate_attn_out", TensorName.attention_out, builder.reshape_0_12),
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, ["test_attention_reshape_k", "test_attention_reshape_v",
                        TensorName.k_cache, TensorName.v_cache, TensorName.slots],
                 [TensorName.k_cache, TensorName.v_cache]),
            call(ANY, ["test_attention_reshape_q", TensorName.k_cache,
                        TensorName.v_cache, TensorName.block_tables, TensorName.seq_len],
                 ["test_attention_intermediate_attn_out"])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_with_compress_type(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        attention_param = {
            'op_name': 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": False,
            "enable_kv_quant": False,
            "atb_attention_param": {"compressType": "COMPRESS_TYPE_KVHEAD"},
            "operation_backend": OperationBackend.ATB,
            "atb_reshape_and_cache_param": {"compressType": "COMPRESS_TYPE_KVHEAD"}
        }
        attention_tensor_map = {
            "q": "q",
            "k": "k",
            "v": "v",
            TensorName.k_cache: TensorName.k_cache,
            TensorName.v_cache: TensorName.v_cache,
            TensorName.slots: TensorName.slots,
            TensorName.block_tables: TensorName.block_tables,
            TensorName.seq_len: TensorName.seq_len,
            TensorName.batch_wins: TensorName.batch_wins,
            TensorName.ra_seq_len: TensorName.ra_seq_len,
            TensorName.attention_out: TensorName.attention_out
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBDecoderPagedAttentionCommonOpBuilder)
        graph = builder.build(graph, attention_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='ReshapeAndCache', op_param='{"compressType": "COMPRESS_TYPE_KVHEAD"}',
                 op_name='test_attention_ReshapeAndCache'),
            call(op_type='PagedAttention', op_param='{"compressType": "COMPRESS_TYPE_KVHEAD"}',
                 op_name='test_attention_PagedAttention')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY,
                 ["test_attention_reshape_k", "test_attention_reshape_v", TensorName.k_cache, TensorName.v_cache,
                  TensorName.slots, TensorName.batch_wins, TensorName.seq_len], 
                  [TensorName.k_cache, TensorName.v_cache]),
            call(ANY, ["test_attention_reshape_q", TensorName.k_cache, 
                        TensorName.v_cache, TensorName.block_tables, TensorName.ra_seq_len],
                 ["test_attention_intermediate_attn_out"])
        ])

    @data(("MASK_TYPE_NORM",), ("MASK_TYPE_ALIBI",))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_with_mask_type(self, mask_type, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        attention_param = {
            'op_name': 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": False,
            "enable_kv_quant": False,
            "atb_attention_param": {"maskType": mask_type},
            "operation_backend": OperationBackend.ATB,
            "atb_reshape_and_cache_param": {}
        }
        attention_tensor_map = {
            "q": "q",
            "k": "k",
            "v": "v",
            TensorName.k_cache: TensorName.k_cache,
            TensorName.v_cache: TensorName.v_cache,
            TensorName.slots: TensorName.slots,
            TensorName.block_tables: TensorName.block_tables,
            TensorName.seq_len: TensorName.seq_len,
            TensorName.attention_mask: TensorName.attention_mask,
            TensorName.attention_out: TensorName.attention_out
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBDecoderPagedAttentionCommonOpBuilder)
        graph = builder.build(graph, attention_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='ReshapeAndCache', op_param='{}', op_name='test_attention_ReshapeAndCache'),
            call(op_type='PagedAttention', op_param=f'{{"maskType": "{mask_type}"}}',
                 op_name='test_attention_PagedAttention')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, ["test_attention_reshape_k", "test_attention_reshape_v",
                       TensorName.k_cache, TensorName.v_cache, TensorName.slots],
                 [TensorName.k_cache, TensorName.v_cache]),
            call(ANY, ["test_attention_reshape_q", TensorName.k_cache, TensorName.v_cache,
                       TensorName.block_tables, TensorName.seq_len, TensorName.attention_mask],
                 ["test_attention_intermediate_attn_out"])
        ])

    @patch("_libatb_torch._BaseOperation")
    def test_build_with_calc_type(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        attention_param = {
            'op_name': 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": False,
            "enable_kv_quant": False,
            "atb_attention_param": {"calcType": "CALC_TYPE_SPEC"},
            "operation_backend": OperationBackend.ATB,
            "atb_reshape_and_cache_param": {}
        }
        attention_tensor_map = {
            "q": "q",
            "k": "k",
            "v": "v",
            TensorName.k_cache: TensorName.k_cache,
            TensorName.v_cache: TensorName.v_cache,
            TensorName.slots: TensorName.slots,
            TensorName.block_tables: TensorName.block_tables,
            TensorName.seq_len: TensorName.seq_len,
            TensorName.q_len: TensorName.q_len,
            TensorName.attention_out: TensorName.attention_out
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBDecoderPagedAttentionCommonOpBuilder)
        graph = builder.build(graph, attention_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='ReshapeAndCache', op_param='{}', op_name='test_attention_ReshapeAndCache'),
            call(op_type='PagedAttention', op_param='{"calcType": "CALC_TYPE_SPEC"}',
                 op_name='test_attention_PagedAttention')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, ["test_attention_reshape_k", "test_attention_reshape_v",
                        TensorName.k_cache, TensorName.v_cache, TensorName.slots],
                 [TensorName.k_cache, TensorName.v_cache]),
            call(ANY, ["test_attention_reshape_q", TensorName.k_cache, TensorName.v_cache,
                        TensorName.block_tables, TensorName.seq_len, TensorName.q_len],
                 ["test_attention_intermediate_attn_out"])
        ])
