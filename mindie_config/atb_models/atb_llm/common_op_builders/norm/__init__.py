# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.norm.norm_common_op_builder import NormCommonOpBuilder
from atb_llm.common_op_builders.norm.quant_norm_common_op_builder import QuantNormCommonOpBuilder


CommonOpBuilderManager.register(NormCommonOpBuilder)
CommonOpBuilderManager.register(QuantNormCommonOpBuilder)