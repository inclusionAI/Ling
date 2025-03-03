# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum 


class OpBackend(int, Enum):
    ATB = 0
    ACLNN = 1
        