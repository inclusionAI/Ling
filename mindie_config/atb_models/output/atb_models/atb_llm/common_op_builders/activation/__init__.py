# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.activation.swiglu_common_op_builder import SwiGLUCommonOpBuilder
from atb_llm.common_op_builders.activation.swish_common_op_builder import SwishCommonOpBuilder


CommonOpBuilderManager.register(SwishCommonOpBuilder)
CommonOpBuilderManager.register(SwiGLUCommonOpBuilder)
