# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.data_type import OperationBackend
from atb_llm.common_op_builders.attention.paged_attention_common_op_builder import PagedAttentionCommonOpBuilder


class ATBDecoderPagedAttentionCommonOpBuilder(PagedAttentionCommonOpBuilder):

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.is_prefill:
            return False
        if self.param.operation_backend != OperationBackend.ATB:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        graph.add_reshape(self.in_tensor_key.q, f"{self.param.op_name}_reshape_q", self.reshape_q)
        graph.add_reshape(self.in_tensor_key.k, f"{self.param.op_name}_reshape_k", self.reshape_kv)
        graph.add_reshape(self.in_tensor_key.v, f"{self.param.op_name}_reshape_v", self.reshape_kv)

        # reshape and cache
        graph = self.add_reshape_and_cache(graph, f"{self.param.op_name}_reshape_k", f"{self.param.op_name}_reshape_v")

        # paged attention
        attention_op = atb._BaseOperation(
            op_type="PagedAttention",
            op_param=json.dumps(self.param.atb_attention_param),
            op_name=f"{self.param.op_name}_PagedAttention"
        )
        graph.operations.append(attention_op)

        input_key_list = [
            f"{self.param.op_name}_reshape_q", self.in_tensor_key.k_cache,
            self.in_tensor_key.v_cache, self.in_tensor_key.block_tables
        ]
        if self.param.atb_attention_param.get("compressType", "COMPRESS_TYPE_UNDEFINED") == "COMPRESS_TYPE_KVHEAD":
            input_key_list.append(self.in_tensor_key.ra_seq_len)
        else:
            input_key_list.append(self.in_tensor_key.seq_len)
        if self.param.atb_attention_param.get("maskType", "UNDEFINED") != "UNDEFINED":
            input_key_list.append(self.in_tensor_key.attention_mask)
        if self.param.atb_attention_param.get("calcType", "CALC_TYPE_UNDEFINED") == "CALC_TYPE_SPEC":
            input_key_list.append(self.in_tensor_key.q_len)
        if self.param.atb_attention_param.get("quantType", "TYPE_QUANT_UNDEFINED") == "TYPE_DEQUANT_FUSION":
            input_key_list.append(self.param.kv_quant_module.k_dequant_scale)
            if self.param.atb_attention_param.get("hasQuantOffset", False):
                input_key_list.append(self.param.kv_quant_module.k_dequant_offset)
            input_key_list.append(self.param.kv_quant_module.v_dequant_scale)
            if self.param.atb_attention_param.get("hasQuantOffset", False):
                input_key_list.append(self.param.kv_quant_module.v_dequant_offset)
        graph.add_operation(attention_op, input_key_list, [f"{self.param.op_name}_intermediate_attn_out"])

        graph.add_reshape(f"{self.param.op_name}_intermediate_attn_out", self.out_tensor_key.attention_out,
                          self.reshape_0_12)

        return graph
