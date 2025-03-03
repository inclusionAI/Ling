# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved
from enum import Enum


class ErrorCode(str, Enum):
    #ATB_MODELS
    ATB_MODELS_PARAM_OUT_OF_RANGE = "MIE05E000000"
    ATB_MODELS_MODEL_PARAM_JSON_INVALID = "MIE05E000001"
    ATB_MODELS_EXECUTION_FAILURE = "MIE05E000002"
    ATB_MODELS_INTERNAL_ERROR = "MIE05E000004"

    def __str__(self):
        return self.value

    


    
