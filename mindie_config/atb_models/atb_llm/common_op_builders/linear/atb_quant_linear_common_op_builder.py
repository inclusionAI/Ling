# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import torch
from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.linear.base_linear_common_op_builder import BaseLinearCommonOpBuilder, \
    BaseLinearCommonOpBuilderParam
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.singleton import Singleton


class ATBQuantLinearCommonOpBuilderParam(BaseLinearCommonOpBuilderParam):
    enable_quant_input: bool = Field(True)
    default_dtype: torch.dtype = Field(torch.float16)


class ATBQuantLinearCommonOpBuilder(BaseLinearCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    @property
    def param_cls(self):
        return ATBQuantLinearCommonOpBuilderParam

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.linear_module.linear_desc not in [LinearTypeV2.W8A8, LinearTypeV2.W8A8S, LinearTypeV2.W8A8SC]:
            return False
        if self.param.linear_module.linear_desc == LinearTypeV2.W8A8SC and self.param.default_dtype == torch.bfloat16:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        if self.param.enable_quant_input:
            # 对input进行量化
            input_quant_op = atb._BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}),
                op_name=self.param.op_name + "_Elewise"
            )
            graph.operations.append(input_quant_op)
            graph.add_operation(
                input_quant_op,
                [self.in_tensor_key.input, f"{self.param.linear_module.prefix}.input_scale",
                 f"{self.param.linear_module.prefix}.input_offset"],
                [self.param.op_name + "_intermediate_input"]
            )

        input_key_list = [
            self.param.op_name + "_intermediate_input" if self.param.enable_quant_input else self.in_tensor_key.input,
            f"{self.param.linear_module.prefix}.weight", f"{self.param.linear_module.prefix}.quant_bias",
            f"{self.param.linear_module.prefix}.deq_scale"
        ]

        if self.param.linear_module.linear_desc == LinearTypeV2.W8A8SC:
            linear_sparse_op = atb._BaseOperation(
                op_type="LinearSparse",
                op_param=json.dumps({"tilingK": 8, "tilingN": 8}),
                op_name=self.param.op_name + "_LinearSparse"
            )
            graph.operations.append(linear_sparse_op)
            graph.add_operation(
                linear_sparse_op,
                [*input_key_list, f"{self.param.linear_module.prefix}.index"],
                [self.out_tensor_key.linear_out]
            )
        else:
            linear_op = atb._BaseOperation(
                op_type="Linear",
                op_param=json.dumps({
                    "transposeB": self.param.linear_module.trans_flag == TransposeType.TRANSPOSE,
                    "outDataType": "ACL_BF16" if self.param.default_dtype == torch.bfloat16 else "ACL_FLOAT16",
                    "hasBias": True}),
                op_name=self.param.op_name + "_LinearQuant"
            )
            graph.operations.append(linear_op)

            graph.add_operation(
                linear_op, input_key_list, [self.out_tensor_key.linear_out]
            )

        return graph
