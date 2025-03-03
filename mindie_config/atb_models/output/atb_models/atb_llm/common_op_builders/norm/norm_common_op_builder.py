# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import _libatb_torch as atb

from atb_llm.common_op_builders.norm.base_norm_common_op_builder import BaseNormCommonOpBuilder
from atb_llm.utils.singleton import Singleton


class NormCommonOpBuilder(BaseNormCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False 
        if self.param.norm_param["normParam"].get("quantType", "QUANT_UNDEFINED") != "QUANT_UNDEFINED":
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict = None) -> atb._GraphOperation:
        input_key_list, output_key_list = super().create_key_list(tensor_map)

        if self.param.has_bias:
            input_key_list.append(self.in_tensor_key.bias)

        graph = super().build_norm(graph, input_key_list, output_key_list)
        
        return graph
