# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import Field
import _libatb_torch as atb

from atb_llm.common_op_builders.norm.base_norm_common_op_builder import BaseNormCommonOpBuilder, \
    BaseNormCommonOpBuilderParam
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.utils.singleton import Singleton


class QuantNormCommonOpBuilderParam(BaseNormCommonOpBuilderParam):
    linear_module: LinearUtils = Field(None)


class QuantNormCommonOpBuilder(BaseNormCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    @property
    def param_cls(self):
        return QuantNormCommonOpBuilderParam
 
    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.norm_param["normParam"].get("quantType", "QUANT_UNDEFINED") != "QUANT_INT8":
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict = None) -> atb._GraphOperation:
        input_key_list, output_key_list = super().create_key_list(tensor_map)

        input_key_list.append(self.in_tensor_key.bias)
        input_key_list.append(f"{self.param.linear_module.prefix}.input_scale")
        input_key_list.append(f"{self.param.linear_module.prefix}.input_offset")
        
        super().build_norm(graph, input_key_list, output_key_list)

        return graph
