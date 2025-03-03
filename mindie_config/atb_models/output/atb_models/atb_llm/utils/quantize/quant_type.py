# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum


class QuantType(str, Enum):
    FLOAT = "float"
    W4A16 = "w4a16"
    W8A8 = "w8a8"
    W8A16 = "w8a16"
    W8A8S = "w8a8s"
    W8A8SC = "w8a8sc"
    W8A8_DYNAMIC = "w8a8_dynamic"


QUANT_W8A8_DESC_LIST = [QuantType.W8A8.upper(), QuantType.W8A8S.upper()]


QUANTIZE_DESC_REQUIRED_LIST = [
    QuantType.W8A8,
    QuantType.W8A8S,
    QuantType.W8A8SC,
    QuantType.W8A16,
    QuantType.W4A16,
    QuantType.W8A8_DYNAMIC
]


class LinearTypeV2(int, Enum):
    INVALID = -1
    FLOAT16 = 0
    BFLOAT16 = 1
    W4A16 = 2
    W8A16 = 3
    W8A8 = 4
    W8A8S = 5
    W8A8SC = 6
    W8A8_DYNAMIC = 7


def is_same_type(linear_desc_list):
    desc_type = None
    for linear_desc in linear_desc_list:
        if desc_type is None:
            desc_type = linear_desc
        if linear_desc != desc_type:
            return False
    return True


def is_all_float(linear_desc_list):
    for linear_desc in linear_desc_list:
        if linear_desc != LinearTypeV2.FLOAT:
            return False
    return True
