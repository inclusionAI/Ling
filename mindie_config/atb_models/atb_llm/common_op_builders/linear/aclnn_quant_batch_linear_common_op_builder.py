# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.linear.base_linear_common_op_builder import BaseLinearCommonOpBuilder, \
    BaseLinearCommonOpBuilderParam
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.singleton import Singleton


class ACLNNQuantBatchLinearCommonOpBuilderParam(BaseLinearCommonOpBuilderParam):
    group_size: int = Field(0)


class ACLNNQuantBatchLinearCommonOpBuilder(BaseLinearCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    @property
    def param_cls(self):
        return ACLNNQuantBatchLinearCommonOpBuilderParam

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.linear_module.linear_desc not in [LinearTypeV2.W4A16, LinearTypeV2.W8A16]:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        input_key_list = [
            self.in_tensor_key.input,
            f"{self.param.linear_module.prefix}.weight", f"{self.param.linear_module.prefix}.weight_scale",
            f"{self.param.linear_module.prefix}.weight_offset"
        ]
        if self.param.linear_module.has_bias:
            input_key_list.append(f"{self.param.linear_module.prefix}.bias")

        op_type = "W8A16MatMul" if self.param.linear_module.linear_desc == LinearTypeV2.W8A16 else "W4A16MatMul"

        linear_op = atb._BaseOperation(
            op_type=op_type,
            op_param=json.dumps({
                "transposeB": self.param.linear_module.trans_flag == TransposeType.TRANSPOSE,
                "quantGroupSize": self.param.group_size,
                "hasBias": self.param.linear_module.has_bias}),
            op_name=self.param.op_name + f"_{op_type}"
        )
        graph.operations.append(linear_op)

        graph.add_operation(
            linear_op, input_key_list, [self.out_tensor_key.linear_out]
        )

        return graph
