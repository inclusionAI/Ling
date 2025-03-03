# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import \
    BaseLinearParallelCommonOpBuilder, CommunicationBackend    
from atb_llm.common_op_builders.linear_parallel.all_reduce_linear_parallel_common_op_builder import \
    AllReduceLinearParallelCommonOpBuilder


class LCOCLinearParallelCommonOpBuilder(AllReduceLinearParallelCommonOpBuilder, BaseLinearParallelCommonOpBuilder):
    def __init__(self):
        super().__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.parallel_info.backend != CommunicationBackend.LCCL:
            return False
        if not self.param.enable_lcoc:
            return False
        linear_module = self.param.linear_param.get("linear_module")
        if isinstance(linear_module, LinearUtils) \
                and linear_module.linear_desc not in [LinearTypeV2.FLOAT16, LinearTypeV2.BFLOAT16]:
            return False
        if self.param.parallel_info.world_size <= 1:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super(BaseLinearParallelCommonOpBuilder, self).build(graph, tensor_map)
        linear_module = self.param.linear_param.get("linear_module")

        linear_parallel_param = self.param.parallel_info.dict()
        linear_parallel_param["transWeight"] = linear_module.trans_flag == TransposeType.TRANSPOSE
        linear_parallel_param["backend"] = "lcoc"

        linear_parallel_op = atb._BaseOperation(
            op_type="LinearParallel",
            op_param=json.dumps(linear_parallel_param),
            op_name=self.param.op_name + "_LinearParallel"
        )
        graph.operations.append(linear_parallel_op)
        graph.add_operation(
            linear_parallel_op,
            [self.in_tensor_key.input, f"{linear_module.prefix}.weight"],
            [self.out_tensor_key.linear_out]
        )

        return graph
