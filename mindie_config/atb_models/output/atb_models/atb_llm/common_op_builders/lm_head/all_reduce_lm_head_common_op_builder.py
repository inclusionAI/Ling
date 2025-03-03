# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.lm_head.base_lm_head_common_op_builder import BaseLmHeadCommonOpBuilder, \
    BaseLmHeadCommonOpBuilderParam
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType


class AllReduceCommonOpBuilderParam(BaseLmHeadCommonOpBuilderParam):
    head_dim: int = Field(...)


class AllReduceLmHeadCommonOpBuilder(BaseLmHeadCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.LM_HEAD

    @property
    def param_cls(self):
        return AllReduceCommonOpBuilderParam

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if not self.param.enable_linear_parallel:
            return False
        parallel_type = self.param.linear_parallel_param.get("parallel_type")
        if parallel_type is not None and parallel_type != ParallelType.ALL_REDUCE:
            return False
        if self.param.parallel_info.world_size <= 1:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        parallel_info_obj = self.param.linear_parallel_param.get("parallel_info")

        if self.param.unpad_inputs:
            op_param = {
                "offsets": [0, self.param.head_dim * parallel_info_obj.rank],
                "size": [-1, self.param.head_dim]
            }
        else:
            op_param = {
                "offsets": [0, 0, self.param.head_dim * parallel_info_obj.rank],
                "size": [-1, -1, self.param.head_dim]
            }

        slice_op = atb._BaseOperation(
            op_type="Slice", 
            op_param=json.dumps(op_param), 
            op_name=f"{self.param.op_name}_Slice"
        )

        graph.operations.append(slice_op)

        in_tensors = [
            self.in_tensor_key.input if not self.param.gather_ahead else f"{self.param.op_name}_intermediate_gather_out"
        ]
        out_tensors = [f"{self.param.op_name}_intermediate_slice_out"]
        graph.add_operation(slice_op, in_tensors, out_tensors)

        builder = CommonOpBuilderManager.get_builder(self.param.linear_parallel_param)
        linear_parallel_tensor_map = {
            "input": f"{self.param.op_name}_intermediate_slice_out",
            "linear_out": self.out_tensor_key.linear_out
        }
        graph = builder.build(graph, linear_parallel_tensor_map)

        return graph
