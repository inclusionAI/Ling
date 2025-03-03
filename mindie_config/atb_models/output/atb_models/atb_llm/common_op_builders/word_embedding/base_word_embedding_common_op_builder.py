# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import TensorParallelInfo
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseWordEmbeddingCommonOpBuilderParam(BaseCommonOpBuilderParam):
    unpad_inputs: bool = Field(False)
    enable_parallel: bool = Field(True)
    parallel_info: TensorParallelInfo = Field(...)


class BaseWordEmbeddingCommonOpBuilderInTensor(BaseModel):
    input_ids: str = Field(...)
    embedding_weights: str = Field(...)


class BaseWordEmbeddingCommonOpBuilderOutTensor(BaseModel):
    word_embedding_out: str = Field(...)


class BaseWordEmbeddingCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.WORD_EMBEDDING

    @property
    def param_cls(self):
        return BaseWordEmbeddingCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseWordEmbeddingCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseWordEmbeddingCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict = None) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)
        return graph
    
    def add_gather(self, graph: atb._GraphOperation, info: str):
        gather_op = atb._BaseOperation(
            op_type="Gather",
            op_param=json.dumps({"axis": 0 if self.param.unpad_inputs else 1}),
            op_name=f"{self.param.op_name}_Gather"
        )
        graph.operations.append(gather_op)
        graph.add_operation(
            gather_op,
            [self.in_tensor_key.embedding_weights, self.in_tensor_key.input_ids],
            [info]
        )
