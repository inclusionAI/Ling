# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from pydantic import BaseModel, Field
import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, NormType


class BaseNormCommonOpBuilderParam(BaseCommonOpBuilderParam):
    has_bias: bool = Field(False)
    enable_add_norm: bool = Field(False)
    norm_type: NormType = Field(None) # LayerNorm or RmsNorm
    norm_param: dict | None = Field({})


class BaseNormCommonOpBuilderInTensor(BaseModel):
    input: str = Field(...)
    weight: str = Field(...)
    bias: str = Field(None)
    residual_input: str = Field(None)
    


class BaseNormCommonOpBuilderOutTensor(BaseModel):
    norm_out: str = Field(...)
    out_add: str = Field(None)


class BaseNormCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.NORM
    
    @property
    def param_cls(self):
        return BaseNormCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseNormCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseNormCommonOpBuilderOutTensor
    
    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        if self.param.norm_param.get("layerType", "RMS_NORM_UNDEFINED").split("_")[-1] == "POSTNORM":
            return False
        return True

    def create_key_list(self, tensor_map: dict = None) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)

        input_key_list = []
        output_key_list = []

        if self.param.enable_add_norm:
            input_key_list.append(self.in_tensor_key.residual_input)

        input_key_list.append(self.in_tensor_key.input)
        input_key_list.append(self.in_tensor_key.weight)

        output_key_list.append(self.out_tensor_key.norm_out)
        if self.param.enable_add_norm:
            output_key_list.append(self.out_tensor_key.out_add)

        return (input_key_list, output_key_list)

    def build_norm(
        self,
        graph: atb._GraphOperation,
        input_key_list: list,
        output_key_list: list
    ) -> atb._GraphOperation:
        norm_op = atb._BaseOperation(
            op_type=self.param.norm_type,
            op_param=json.dumps(self.param.norm_param),
            op_name=self.param.op_name
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op,
            input_key_list,
            output_key_list
        )
        return graph