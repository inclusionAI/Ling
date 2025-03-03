# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseGateUpCommonOpBuilderParam(BaseCommonOpBuilderParam):
    is_pack: bool = Field(True)
    linear_param: dict | None = Field(None)


class BaseGateUpCommonOpBuilderInTensor(BaseModel):
    input: str = Field(...)


class BaseGateUpCommonOpBuilderOutTensor(BaseModel):
    gate_up_out: str = Field(...)


class BaseGateUpCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.GATE_UP

    @property
    def param_cls(self):
        return BaseGateUpCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseGateUpCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseGateUpCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)
        return graph