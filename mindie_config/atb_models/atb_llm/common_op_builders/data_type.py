# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum


class CommonOpBuilderType(str, Enum):
    BASE = "BASE"
    LINEAR = "LINEAR"
    LINEAR_PARALLEL = "LINEAR_PARALLEL"
    WORD_EMBEDDING = "WORD_EMBEDDING"
    POSITIONAL_EMBEDDING = "POSITIONAL_EMBEDDING"
    NORM = "NORM"
    QKV = "QKV"
    ATTENTION = "ATTENTION"
    LM_HEAD = "LM_HEAD"
    GATE_UP = "GATE_UP"
    ACTIVATION = "ACTIVATION"
    ROPE = "ROPE"


class NormType(str, Enum):
    RMSNORM = "RmsNorm"
    LAYERNORM = "LayerNorm"


class CommonOpBuilderOwner(str, Enum):
    DEFAULT = "DEFAULT"


class ActivationType(str, Enum):
    SWIGLU = "swiglu"
    SWISH = "swish"


class OperationBackend(str, Enum):
    ATB = "ATB"
    ACLNN = "ACLNN"
