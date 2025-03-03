# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import _libatb_torch as atb

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.lm_head.base_lm_head_common_op_builder import BaseLmHeadCommonOpBuilder
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class NoParallelLmHeadCommonOpBuilder(BaseLmHeadCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.LM_HEAD

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.enable_linear_parallel and self.param.linear_parallel_param.get("parallel_info").world_size > 1:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        builder = CommonOpBuilderManager.get_builder(self.param.linear_parallel_param.get("linear_param", {}))
        in_key = f"{self.param.op_name}_intermediate_gather_out" if self.param.gather_ahead \
            else self.in_tensor_key.input
        linear_tensor_map = {
            "input": in_key,
            "linear_out": self.out_tensor_key.linear_out
        }
        graph = builder.build(graph, linear_tensor_map)

        return graph
