# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.log import logger, print_log
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseRopeCommonOpBuilderParam(BaseCommonOpBuilderParam):
    is_fa: bool = Field(...)
    head_num: int = Field(...)
    kv_head_num: int = Field(...)
    atb_rope_param: dict | None = Field({})


class BaseRopeCommonOpBuilderInTensor(BaseModel):
    q: str = Field(...)
    k: str = Field(...)
    cos_embedding: str = Field(...)
    sin_embedding: str = Field(...)
    seq_len: str = Field(...)


class BaseRopeCommonOpBuilderOutTensor(BaseModel):
    q_out: str = Field(...)
    k_out: str = Field(...)


class BaseRopeCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.ROPE

    @property
    def param_cls(self):
        return BaseRopeCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseRopeCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseRopeCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict = None) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)
        return graph
