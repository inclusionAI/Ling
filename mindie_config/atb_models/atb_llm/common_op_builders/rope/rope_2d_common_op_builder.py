# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.rope.base_rope_common_op_builder import BaseRopeCommonOpBuilder
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.singleton import Singleton


class Rope2dCommonOpBuilder(BaseRopeCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def reshape_q(self, org_shape):
        return [org_shape[0], org_shape[1], self.param.head_num, org_shape[2] // self.param.head_num]
    
    def reshape_kv(self, org_shape):
        return [org_shape[0], org_shape[1], self.param.kv_head_num, org_shape[2] // self.param.kv_head_num]
    
    def reshape_01_2(self, org_shape):
        return [org_shape[0] * org_shape[1], org_shape[2]]

    def build(self, graph: atb._GraphOperation, tensor_map) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        input_key_list = []
        if self.param.is_fa:
            graph.add_reshape(self.in_tensor_key.q, f"{self.param.op_name}_q", self.reshape_q)
            graph.add_reshape(self.in_tensor_key.k, f"{self.param.op_name}_k", self.reshape_kv)
            graph.add_reshape(self.in_tensor_key.cos_embedding, f"{self.param.op_name}_cos", self.reshape_01_2)
            graph.add_reshape(self.in_tensor_key.sin_embedding, f"{self.param.op_name}_sin", self.reshape_01_2)
            input_key_list.extend([
                f"{self.param.op_name}_q", f"{self.param.op_name}_k", 
                f"{self.param.op_name}_cos", f"{self.param.op_name}_sin"
            ])
        else:
            input_key_list.extend([
                self.in_tensor_key.q, self.in_tensor_key.k, 
                self.in_tensor_key.cos_embedding, self.in_tensor_key.sin_embedding
            ])
        input_key_list.append(self.in_tensor_key.seq_len)

        rope_op = atb._BaseOperation(
            op_type="Rope", 
            op_param=json.dumps(self.param.atb_rope_param), 
            op_name=f"{self.param.op_name}_Rope"
        )
        graph.operations.append(rope_op)

        graph.add_operation(
            rope_op,
            input_key_list,
            [self.out_tensor_key.q_out, self.out_tensor_key.k_out]
        )

        return graph
