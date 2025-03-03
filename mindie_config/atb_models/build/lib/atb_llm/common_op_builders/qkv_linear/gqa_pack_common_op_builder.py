# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.qkv_linear.base_qkv_linear_common_op_builder import BaseQKVLinearCommonOpBuilder
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.utils.singleton import Singleton


class GqaPackCommonOpBuilder(BaseQKVLinearCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        if self.param.head_num == self.param.kv_head_num:
            return False
        if not self.param.is_pack:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)
        offsets_key = "offsets"
        size_key = "size"
        # add qkv linear
        self.param.linear_param.update({"linear_module": self.param.linear_modules[0]})
        qkv_linear_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        qkv_linear_tensor_map = {
            "input": self.in_tensor_key.input,
            "linear_out": f"{self.param.op_name}_intermediate_mixed_qkv"
        }
        graph = qkv_linear_builder.build(graph, qkv_linear_tensor_map)
        # add slice q
        slice_q_param = {}
        if self.param.is_fa:
            slice_q_param[offsets_key] = [0, 0, 0]
            slice_q_param[size_key] = [-1, -1, self.param.head_num * self.param.head_dim]
        else:
            slice_q_param[offsets_key] = [0, 0]
            slice_q_param[size_key] = [-1, self.param.head_num * self.param.head_dim]
        slice_q_op = atb._BaseOperation(
            op_type="Slice",
            op_param=json.dumps(slice_q_param),
            op_name=self.param.op_name + "_Slice_Q"
        )
        graph.operations.append(slice_q_op)
        graph.add_operation(
            slice_q_op,
            [f"{self.param.op_name}_intermediate_mixed_qkv"],
            [self.out_tensor_key.q_out],
        )
        # add slice kv
        slice_kv_param = {}
        if self.param.is_fa:
            slice_kv_param[offsets_key] = [0, 0, self.param.head_num * self.param.head_dim]
            slice_kv_param[size_key] = [-1, -1, self.param.kv_head_num * self.param.head_dim * 2]
        else:
            slice_kv_param[offsets_key] = [0, self.param.head_num * self.param.head_dim]
            slice_kv_param[size_key] = [-1, self.param.kv_head_num * self.param.head_dim * 2]
        slice_kv_op = atb._BaseOperation(
            op_type="Slice",
            op_param=json.dumps(slice_kv_param),
            op_name=self.param.op_name + "_Slice_KV"
        )
        graph.operations.append(slice_kv_op)
        graph.add_operation(
            slice_kv_op,
            [f"{self.param.op_name}_intermediate_mixed_qkv"],
            [f"{self.param.op_name}_intermediate_mixed_kv"],
        )
        # add split kv
        split_op = atb._BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 2}),
            op_name=self.param.op_name + "_Split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            [f"{self.param.op_name}_intermediate_mixed_kv"],
            [self.out_tensor_key.k_out, self.out_tensor_key.v_out],
        )
        
        return graph
