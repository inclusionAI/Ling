# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from abc import abstractmethod

from pydantic import BaseModel, Field, error_wrappers

from atb_llm.common_op_builders.data_type import CommonOpBuilderType, CommonOpBuilderOwner
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.env import ENV
import _libatb_torch as atb


class BaseCommonOpBuilderParam(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    op_name: str = Field("")
    category: CommonOpBuilderType = Field(CommonOpBuilderType.BASE)
    owner: CommonOpBuilderOwner = Field(CommonOpBuilderOwner.DEFAULT)


class BaseCommonOpBuilder:
    def __init__(self):
        self.common_op_builder_name = self.__class__.__name__
        self.category = CommonOpBuilderType.BASE
        self.owner = CommonOpBuilderOwner.DEFAULT
        self.param = None
        self.in_tensor_key = None
        self.out_tensor_key = None

    @property
    @abstractmethod
    def param_cls(self):
        raise NotImplementedError(
            f"CommonOpBuilder {self.common_op_builder_name} param_cls property is not implemented")

    @property
    @abstractmethod
    def in_tensor_cls(self):
        raise NotImplementedError(
            f"CommonOpBuilder {self.common_op_builder_name} in_tensor_cls property is not implemented")

    @property
    @abstractmethod
    def out_tensor_cls(self):
        raise NotImplementedError(
            f"CommonOpBuilder {self.common_op_builder_name} out_tensor_cls property is not implemented")

    def verify_base_param(self, param: dict) -> bool:
        try:
            base_param_obj = BaseCommonOpBuilderParam(**param)
        except error_wrappers.ValidationError as e:
            print_log(ENV.rank, logger.info,
                      f"CommonOpBuilder {self.common_op_builder_name} parameter validation error: {e}")
            return False

        if base_param_obj.category != self.category:
            return False
        if base_param_obj.owner != self.owner:
            print_log(ENV.rank, logger.info,
                      f"CommonOpBuilder {self.common_op_builder_name} owner doesn't match, "
                      f"looking for {self.owner} "
                      f"but got {base_param_obj.owner} instead")
            return False

        try:
            param_obj = self.param_cls(**param)
        except error_wrappers.ValidationError as e:
            print_log(ENV.rank, logger.info,
                      f"CommonOpBuilder {self.common_op_builder_name} parameter validation error: {e}")
            return False
        self.param = param_obj
        return True

    def is_match(self, param: dict) -> bool:
        if not self.verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.model_validate(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.model_validate(tensor_map)
        return graph
