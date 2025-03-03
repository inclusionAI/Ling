# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from enum import Enum

from pydantic import BaseModel, Field

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.linear.base_linear_common_op_builder import BaseLinearCommonOpBuilderInTensor, \
    BaseLinearCommonOpBuilderOutTensor
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class CommunicationBackend(str, Enum):
    LCCL = "lccl"
    HCCL = "hccl"


class TensorParallelInfo(BaseModel):
    rank: int = Field(0)
    world_size: int = Field(1)
    backend: CommunicationBackend = Field(CommunicationBackend.HCCL)
    rank_table_file: str = Field("")

    def json(self, **kwargs):
        ori_json = super().json(**kwargs)
        ori_json = json.loads(ori_json)
        ret = {}
        ret["rank"] = ori_json["rank"]
        ret["rankSize"] = ori_json["world_size"]
        ret["backend"] = ori_json["backend"]
        ret["rankTableFile"] = ori_json["rank_table_file"]
        return json.dumps(ret)

    def dict(self, **kwargs):
        ori_dict = super().dict(**kwargs)
        ret = {}
        ret["rank"] = ori_dict["rank"]
        ret["rankSize"] = ori_dict["world_size"]
        ret["backend"] = ori_dict["backend"]
        ret["rankTableFile"] = ori_dict["rank_table_file"]
        return ret


class ParallelType(str, Enum):
    ALL_GATHER = "ALL_GATHER"
    ALL_REDUCE = "ALL_REDUCE"


class BaseLinearParallelCommonOpBuilderParam(BaseCommonOpBuilderParam):
    parallel_type: ParallelType = Field(...)
    parallel_info: TensorParallelInfo = Field(...)
    linear_param: dict = Field(...)


class BaseLinearParallelCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.LINEAR_PARALLEL

    @property
    def param_cls(self):
        return BaseLinearParallelCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseLinearCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseLinearCommonOpBuilderOutTensor
