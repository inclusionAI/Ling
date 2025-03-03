# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.linear.base_linear_common_op_builder import BaseLinearCommonOpBuilderInTensor, \
    BaseLinearCommonOpBuilderOutTensor
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseLmHeadCommonOpBuilderParam(BaseCommonOpBuilderParam):
    gather_ahead: bool = Field(False)
    unpad_inputs: bool = Field(False)
    enable_linear_parallel: bool = Field(True)
    linear_parallel_param: dict = Field(...)


class BaseLmHeadCommonOpBuilderInTensor(BaseLinearCommonOpBuilderInTensor):
    indices: str = Field("")


class BaseLmHeadCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.LM_HEAD

    @property
    def param_cls(self):
        return BaseLmHeadCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseLmHeadCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseLinearCommonOpBuilderOutTensor

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.model_validate(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.model_validate(tensor_map)

        if self.param.gather_ahead:
            gather_op = atb._BaseOperation(
                op_type="Gather",
                op_param=json.dumps({
                    "axis": 0 if self.param.unpad_inputs else 1
                }),
                op_name=f"{self.param.op_name}_Gather"
            )
            graph.operations.append(gather_op)

            input_tensors = [self.in_tensor_key.input, self.in_tensor_key.indices]
            output_tensors = [f"{self.param.op_name}_intermediate_gather_out"]

            graph.add_operation(gather_op, input_tensors, output_tensors)

        return graph
