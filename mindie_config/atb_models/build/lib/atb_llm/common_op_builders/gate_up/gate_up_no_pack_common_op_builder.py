# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.gate_up.base_gate_up_common_op_builder import \
    BaseGateUpCommonOpBuilder, BaseGateUpCommonOpBuilderParam, BaseGateUpCommonOpBuilderOutTensor
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.utils.singleton import Singleton


class GateUpNoPackCommonOpBuilderParam(BaseGateUpCommonOpBuilderParam):
    up_linear_param: dict | None = Field(None)


class GateUpNoPackCommonOpBuilderOutTensor(BaseGateUpCommonOpBuilderOutTensor):
    up_out: str = Field(...)


class GateUpNoPackCommonOpBuilder(BaseGateUpCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()

    @property
    def param_cls(self):
        return GateUpNoPackCommonOpBuilderParam

    @property
    def out_tensor_cls(self):
        return GateUpNoPackCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.is_pack:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        gate_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        gate_tensor_map = {
            "input": self.in_tensor_key.input,
            "linear_out": self.out_tensor_key.gate_up_out
        }
        graph = gate_builder.build(graph, gate_tensor_map)
        
        up_builder = CommonOpBuilderManager.get_builder(self.param.up_linear_param)
        up_tensor_map = {
            "input": self.in_tensor_key.input,
            "linear_out": self.out_tensor_key.up_out
        }
        graph = up_builder.build(graph, up_tensor_map)

        return graph