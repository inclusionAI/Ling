# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.attention.base_attention_common_op_builder import BaseAttentionCommonOpBuilder, \
    BaseAttentionCommonOpBuilderParam, BaseAttentionCommonOpBuilderInTensor, AttnType


class PagedAttentionCommonOpBuilderParam(BaseAttentionCommonOpBuilderParam):
    atb_reshape_and_cache_param: dict = Field({})


class PagedAttentionCommonOpBuilderInTensor(BaseAttentionCommonOpBuilderInTensor):
    slots: str = Field(...)
    block_tables: str = Field(...)
    q_len: str | None = Field(None)
    ra_seq_len: str | None = Field(None)
    batch_wins: str | None = Field(None)


class PagedAttentionCommonOpBuilder(BaseAttentionCommonOpBuilder):
    def __init__(self):
        super().__init__()

    @property
    def param_cls(self):
        return PagedAttentionCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return PagedAttentionCommonOpBuilderInTensor

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.attn_type != AttnType.PAGED_ATTENTION:
            return False
        return True

    def add_kv_quant(self, graph: atb._GraphOperation, in_tensor_key: str, target_key: str) -> atb._GraphOperation:
        elewise_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}),
            op_name=f"{self.param.op_name}_{target_key}_Elewise"
        )
        graph.operations.append(elewise_op)
        graph.add_operation(
            elewise_op,
            [in_tensor_key,
             f"{self.param.kv_quant_module.prefix}.{target_key}_quant_scale",
             f"{self.param.kv_quant_module.prefix}.{target_key}_quant_offset"],
            [f"{self.param.op_name}_intermediate_{target_key}_int8"]
        )
        return graph

    def add_reshape_and_cache(
            self, graph: atb._GraphOperation, k_tensor_key: str, v_tensor_key: str) -> atb._GraphOperation:
        # KV int 8
        if self.param.enable_kv_quant:
            graph = self.add_kv_quant(graph, k_tensor_key, "k")
            graph = self.add_kv_quant(graph, v_tensor_key, "v")

        # reshape and cache
        reshape_and_cache_op = atb._BaseOperation(
            op_type="ReshapeAndCache",
            op_param=json.dumps(self.param.atb_reshape_and_cache_param),
            op_name=f"{self.param.op_name}_ReshapeAndCache"
        )

        input_key_list = []
        if self.param.enable_kv_quant:
            input_key_list.extend(
                [f"{self.param.op_name}_intermediate_k_int8", f"{self.param.op_name}_intermediate_v_int8"])
        else:
            input_key_list.extend([k_tensor_key, v_tensor_key])
        input_key_list.extend([self.in_tensor_key.k_cache, self.in_tensor_key.v_cache, self.in_tensor_key.slots])
        if self.param.atb_reshape_and_cache_param.get("compressType",
                                                      'COMPRESS_TYPE_UNDEFINED') == 'COMPRESS_TYPE_KVHEAD':
            input_key_list.extend([self.in_tensor_key.batch_wins, self.in_tensor_key.seq_len])

        graph.operations.append(reshape_and_cache_op)
        graph.add_operation(
            reshape_and_cache_op,
            input_key_list,
            [self.in_tensor_key.k_cache, self.in_tensor_key.v_cache]
        )
        return graph
