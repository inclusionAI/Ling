# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, ActivationType


class BaseActivationCommonOpBuilderParam(BaseCommonOpBuilderParam):
    is_pack: bool = Field(True)
    up_weight_only: bool = Field(False)
    activation_type: ActivationType = Field(ActivationType.SWIGLU)


class BaseActivationCommonOpBuilderInTensor(BaseModel):
    input: str = Field(...)
    other_input: str = Field(None)


class BaseActivationCommonOpBuilderOutTensor(BaseModel):
    act_out: str = Field(...)


class BaseActivationCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.ACTIVATION

    @property
    def param_cls(self):
        return BaseActivationCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseActivationCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseActivationCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)
        return graph