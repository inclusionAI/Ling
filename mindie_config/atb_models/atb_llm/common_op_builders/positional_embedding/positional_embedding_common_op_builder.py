# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.utils.singleton import Singleton


class PositionalEmbeddingCommonOpBuilderInTensor(BaseModel):
    position_ids: str = Field(..., description="Position IDs")
    cos_table: str = Field(..., description="Cos table")
    sin_table: str = Field(..., description="Sin table")


class PositionalEmbeddingCommonOpBuilderOutTensor(BaseModel):
    cos_embedding: str = Field(..., description="Cos embedding")
    sin_embedding: str = Field(..., description="Sin embedding")


class PositionalEmbeddingCommonOpBuilder(BaseCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.POSITIONAL_EMBEDDING

    @property
    def param_cls(self):
        return BaseCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return PositionalEmbeddingCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return PositionalEmbeddingCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)

        cos_table_gather_op = atb._BaseOperation(
            op_type="Gather",
            op_param="{}",
            op_name=f"{self.param.op_name}_Gather_cosine_table"
        )

        sine_table_gather_op = atb._BaseOperation(
            op_type="Gather",
            op_param="{}",
            op_name=f"{self.param.op_name}_Gather_sine_table"
        )

        graph.operations.extend([cos_table_gather_op, sine_table_gather_op])

        graph.add_operation(
            cos_table_gather_op,
            [self.in_tensor_key.cos_table, self.in_tensor_key.position_ids],
            [self.out_tensor_key.cos_embedding]
        )

        graph.add_operation(
            sine_table_gather_op,
            [self.in_tensor_key.sin_table, self.in_tensor_key.position_ids],
            [self.out_tensor_key.sin_embedding]
        )

        return graph
