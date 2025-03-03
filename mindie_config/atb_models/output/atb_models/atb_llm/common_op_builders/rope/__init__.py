# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.rope.rope_2d_common_op_builder import Rope2dCommonOpBuilder


CommonOpBuilderManager.register(Rope2dCommonOpBuilder)