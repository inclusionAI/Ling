# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.linear.base_linear_common_op_builder import BaseLinearCommonOpBuilderInTensor
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.log import logger, print_log
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseQKVLinearCommonOpBuilderParam(BaseCommonOpBuilderParam):
    is_pack: bool = Field(False)
    is_fa: bool = Field(False)
    head_dim: int = Field(...)
    head_num: int = Field(...)
    kv_head_num: int = Field(...)
    linear_modules: list | None = Field({})
    linear_param: dict | None = Field({})


class BaseQKVLinearCommonOpBuilderOutTensor(BaseModel):
    q_out: str = Field(...)
    k_out: str = Field(...)
    v_out: str = Field(...)


class BaseQKVLinearCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.QKV

    @property
    def param_cls(self):
        return BaseQKVLinearCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseLinearCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseQKVLinearCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)
        return graph
