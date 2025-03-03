# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum

from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, OperationBackend
from atb_llm.utils.layers.attention.kv_cache import KvCache


class AttnType(str, Enum):
    PAGED_ATTENTION = "PAGED_ATTENTION"
    FLASH_ATTENTION = "FLASH_ATTENTION"


class BaseAttentionCommonOpBuilderParam(BaseCommonOpBuilderParam):
    attn_type: AttnType = Field(AttnType.PAGED_ATTENTION)
    is_prefill: bool = Field(False)
    enable_kv_quant: bool = Field(False)
    kv_quant_module: KvCache | None = Field(None)
    head_size: int = Field(128)
    atb_attention_param: dict | None = Field({})
    operation_backend: OperationBackend = Field(OperationBackend.ATB)


class BaseAttentionCommonOpBuilderInTensor(BaseModel):
    q: str = Field(...)
    k: str = Field(...)
    v: str = Field(...)
    k_cache: str = Field(...)
    v_cache: str = Field(...)
    attention_mask: str | None = Field(None)
    seq_len: str | None = Field(None)
    slopes: str | None = Field(None)


class BaseAttentionCommonOpBuilderOutTensor(BaseModel):
    attention_out: str = Field(...)


class BaseAttentionCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.ATTENTION

    @property
    def param_cls(self):
        return BaseAttentionCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseAttentionCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseAttentionCommonOpBuilderOutTensor

    def reshape_q(self, org_shape):
        return [org_shape[0], self.param.atb_attention_param.get("headNum", 0), self.param.head_size]

    def reshape_kv(self, org_shape):
        return [org_shape[0], self.param.atb_attention_param.get("kvHeadNum", 0), self.param.head_size]

    def reshape_0_12(self, org_shape):
        return [org_shape[0], org_shape[1] * org_shape[2]]

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.model_validate(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.model_validate(tensor_map)

        return graph
