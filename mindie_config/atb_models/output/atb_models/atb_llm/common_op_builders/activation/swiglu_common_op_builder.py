# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.activation.base_activation_common_op_builder import BaseActivationCommonOpBuilder
from atb_llm.common_op_builders.data_type import ActivationType
from atb_llm.utils.singleton import Singleton


class SwiGLUCommonOpBuilder(BaseActivationCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
    
    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.activation_type != ActivationType.SWIGLU:
            return False
        if self.param.up_weight_only:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)
        
        if not self.param.is_pack:
            concat_op = atb._BaseOperation(
                op_type="Concat",
                op_param=json.dumps({"concatDim": -1}),
                op_name=self.param.op_name + "_Concat"
            )
            graph.operations.append(concat_op)
            graph.add_operation(
                concat_op,
                [self.in_tensor_key.input, self.in_tensor_key.other_input],
                [self.param.op_name + "_intermidiate_gate_up"]
            )

        swiglu_op = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_SWIGLU_FORWARD', "dim": -1}),
            op_name=self.param.op_name + '_Swiglu'
        )
        graph.operations.append(swiglu_op)
        graph.add_operation(
            swiglu_op,
            [self.in_tensor_key.input if self.param.is_pack else self.param.op_name + "_intermidiate_gate_up"],
            [self.out_tensor_key.act_out],
        )
        return graph