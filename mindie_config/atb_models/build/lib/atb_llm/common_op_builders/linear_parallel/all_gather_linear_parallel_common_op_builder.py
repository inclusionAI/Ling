# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import \
    BaseLinearParallelCommonOpBuilder, ParallelType, BaseLinearParallelCommonOpBuilderParam
from atb_llm.utils.singleton import Singleton


class AllGatherLinearParallelCommonOpBuilderParam(BaseLinearParallelCommonOpBuilderParam):
    unpad_inputs: bool = True


class AllGatherLinearParallelCommonOpBuilder(BaseLinearParallelCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    @property
    def param_cls(self):
        return AllGatherLinearParallelCommonOpBuilderParam

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.parallel_type != ParallelType.ALL_GATHER:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)
        builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        linear_tensor_map = {
            "input": self.in_tensor_key.input,
            "linear_out": self.out_tensor_key.linear_out
            if self.param.parallel_info.world_size <= 1 else self.param.op_name + "_intermediate_linear_out"
        }
        graph = builder.build(graph, linear_tensor_map)

        if self.param.parallel_info.world_size <= 1:
            return graph

        all_gather_op = atb._BaseOperation(
            op_type="AllGather",
            op_param=self.param.parallel_info.json(),
            op_name=self.param.op_name + "_AllGather",
        )
        graph.operations.append(all_gather_op)
        graph.add_operation(
            all_gather_op,
            [self.param.op_name + "_intermediate_linear_out"],
            [self.param.op_name + "_all_gather_out"]
        )
        transpose_op = atb._BaseOperation(
            op_type="Transpose",
            op_param=json.dumps({"perm": [1, 0, 2] if self.param.unpad_inputs else [1, 2, 0, 3]}),
            op_name=self.param.op_name + "_Transpose",
        )
        graph.operations.append(transpose_op)
        graph.add_operation(
            transpose_op,
            [self.param.op_name + "_all_gather_out"],
            [self.out_tensor_key.linear_out]
        )

        return graph
