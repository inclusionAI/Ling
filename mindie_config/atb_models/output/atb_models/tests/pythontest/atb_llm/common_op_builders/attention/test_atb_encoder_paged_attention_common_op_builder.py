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
from atb_llm.common_op_builders.attention.atb_encoder_paged_attention_common_op_builder import \
    ATBEncoderPagedAttentionCommonOpBuilder
from atb_llm.common_op_builders.data_type import OperationBackend
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.layers.attention.kv_cache import KvCache
from tests.pythontest.atb_llm.common_op_builders.attention.test_atb_decoder_paged_attention_common_op_builder import \
    TensorName


@ddt
class TestATBEncoderPagedAttentionCommonOpBuilder(unittest.TestCase):
    @data((True,), (False,))
    @unpack
    def test_is_match(self, enable_kv_quant):
        ATBEncoderPagedAttentionCommonOpBuilder().build = Mock()
        kv_quant_module = KvCache(torch.rand([10]), torch.rand([10]), torch.float16)
        attention_param = {
            'op_name': 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": True,
            "enable_kv_quant": enable_kv_quant,
            "kv_quant_module": kv_quant_module,
            "atb_attention_param": {"calcType": "PA_ENCODER"},
            "operation_backend": OperationBackend.ATB,
            "atb_reshape_and_cache_param": {}
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBEncoderPagedAttentionCommonOpBuilder)

    @patch("_libatb_torch._BaseOperation")
    def test_build_kv_quant_enable_false(self, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        graph.add_reshape = Mock()
        attention_param = {
            "op_name": "test_attention",
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": True,
            "enable_kv_quant": False,
            "atb_attention_param": {"calcType": "PA_ENCODER"},
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
        self.assertIsInstance(builder, ATBEncoderPagedAttentionCommonOpBuilder)
        graph = builder.build(graph, attention_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='ReshapeAndCache', op_param='{}', op_name='test_attention_ReshapeAndCache'),
            call(op_type='SelfAttention', op_param='{"calcType": "PA_ENCODER"}', op_name='test_attention_SelfAttention')
        ])
        graph.add_reshape.assert_has_calls([
            call("q", TensorName.test_reshape_q, builder.reshape_q),
            call("k", TensorName.test_reshape_k, builder.reshape_kv),
            call("v", TensorName.test_reshape_v, builder.reshape_kv),
            call("test_attention_intermediate_attn_out", TensorName.attention_out, builder.reshape_0_12),
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, [TensorName.test_reshape_k, TensorName.test_reshape_v, 
                       TensorName.k_cache, TensorName.v_cache, TensorName.slots],
                 [TensorName.k_cache, TensorName.v_cache]),
            call(ANY, [TensorName.test_reshape_q, TensorName.test_reshape_k, 
                       TensorName.test_reshape_v, TensorName.seq_len],
                 ["test_attention_intermediate_attn_out"])
        ])

    @data(("MASK_TYPE_ALIBI_COMPRESS",), ("MASK_TYPE_ALIBI_COMPRESS_SQRT",), ("MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN",))
    @unpack
    @patch("_libatb_torch._BaseOperation")
    def test_build_with_mask_free(self, mask_type, mock_atb_operation):
        graph = AtbGraph("graph")
        graph.add_operation = Mock()
        attention_param = {
            "op_name": 'test_attention',
            "common_op_builder_category": CommonOpBuilderType.ATTENTION,
            "attn_type": AttnType.PAGED_ATTENTION,
            "is_prefill": True,
            "enable_kv_quant": False,
            "atb_attention_param": {"calcType": "PA_ENCODER", "maskType": mask_type},
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
            TensorName.slopes: TensorName.slopes,
            TensorName.attention_out: TensorName.attention_out
        }
        builder = CommonOpBuilderManager.get_builder(attention_param)
        self.assertIsInstance(builder, ATBEncoderPagedAttentionCommonOpBuilder)
        graph = builder.build(graph, attention_tensor_map)
        mock_atb_operation.assert_has_calls([
            call(op_type='ReshapeAndCache', op_param='{}', op_name='test_attention_ReshapeAndCache'),
            call(op_type='SelfAttention', op_param=f'{{"calcType": "PA_ENCODER", "maskType": "{mask_type}"}}',
                 op_name='test_attention_SelfAttention')
        ])
        graph.add_operation.assert_has_calls([
            call(ANY, [TensorName.test_reshape_k, TensorName.test_reshape_v, 
                       TensorName.k_cache, TensorName.v_cache, TensorName.slots],
                 [TensorName.k_cache, TensorName.v_cache]),
            call(ANY, [TensorName.test_reshape_q, TensorName.test_reshape_k, TensorName.test_reshape_v,
                       TensorName.attention_mask, TensorName.seq_len, TensorName.slopes],
                 ["test_attention_intermediate_attn_out"])
        ])
