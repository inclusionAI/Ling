# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.activation.base_activation_common_op_builder import BaseActivationCommonOpBuilder
from atb_llm.common_op_builders.data_type import ActivationType
from atb_llm.utils.singleton import Singleton


class SwishCommonOpBuilder(BaseActivationCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.activation_type != ActivationType.SWISH:
            return False
        if self.param.up_weight_only and not self.param.is_pack:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)
        
        if self.param.is_pack:
            split_op = atb._BaseOperation(
                op_type="Split",
                op_param=json.dumps({
                    "splitDim": -1,
                    "splitNum": 2
                }),
                op_name=self.param.op_name + "_Split"
            )
            graph.operations.append(split_op)
            graph.add_operation(
                split_op,
                [self.in_tensor_key.input],
                [self.param.op_name + "_intermediate_gate", self.param.op_name + "_intermediate_up"],
            )
        swish_input_list = []
        if self.param.is_pack:
            swish_input_list.append(self.param.op_name + "_intermediate_gate")
        else:
            swish_input_list.append(self.in_tensor_key.input)
        swish_output_list = []
        if self.param.up_weight_only:
            swish_output_list.append(self.out_tensor_key.act_out)
        else:
            swish_output_list.append(self.param.op_name + "_intermediate_swish")

        swish_op = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_SWISH', "dim": -1}),
            op_name=self.param.op_name + "_Swish"
        )
        graph.operations.append(swish_op)
        graph.add_operation(
            swish_op,
            swish_input_list,
            swish_output_list,
        )

        if not self.param.up_weight_only:
            matmul_op = atb._BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({
                    'elewiseType': 'ELEWISE_MUL'}),
                op_name=self.param.op_name + "_Matmul"
            )
            graph.operations.append(matmul_op)
            matmul_input_list = [self.param.op_name + "_intermediate_swish"]
            if self.param.is_pack:
                matmul_input_list.append(self.param.op_name + "_intermediate_up")
            else:
                matmul_input_list.append(self.in_tensor_key.other_input)
            graph.add_operation(
                matmul_op,
                matmul_input_list,
                [self.out_tensor_key.act_out],
            )

        return graph