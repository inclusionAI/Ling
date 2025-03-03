# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum
from .quant_type import QuantType, QUANT_W8A8_DESC_LIST


class PackType(int, Enum):
    ALL_FP = 1
    ALL_W8A8 = 2
    ALL_W8A8_ANTI = 3
    MIX_W8A8 = 4
    MIX_W8A8_ANTI = 5
    ALL_W8A16 = 6
    ALL_W8A8SC = 7
    MIX_W8A8SC = 8
    ALL_W8A8SC_ANTI = 9
    MIX_W8A8SC_ANTI = 10
    ALL_W4A16 = 11
    ALL_W8A16_ANTI = 12
    ALL_W4A16_ANTI = 13
    MIX_W4A16 = 14
    MIX_W4A16_ANTI = 15
    MIX_W8A16 = 16
    MIX_W8A16_ANTI = 17
    ALL_W8A8_DYNAMIC = 18
    ALL_W8A8_DYNAMIC_ANTI = 19
    MIX_W8A8_DYNAMIC = 20
    MIX_W8A8_DYNAMIC_ANTI = 21


def is_w8a8sc(type_desc):
    if type_desc == QuantType.W8A8SC.upper():
        return True
    else:
        return False


def calc_w8a8sc_linear_pack_type(weights, linear_names, norm_name=None, pack_name=None):
    norm_anti_desc = f'{norm_name}.anti.weight'
    is_anti = True if norm_anti_desc in weights.quant_desc else False

    quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
    if quant_desc is not None:
        if quant_desc == QuantType.W8A8SC.upper() and is_anti:
            return PackType.ALL_W8A8SC_ANTI
        elif quant_desc == QuantType.W8A8SC.upper():
            return PackType.ALL_W8A8SC
        elif quant_desc == QuantType.FLOAT.upper():
            return PackType.ALL_FP

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    is_w8a8sc_list = [is_w8a8sc(linear_desc) for linear_desc in linear_desces]

    is_any_w8a8sc = any(is_w8a8sc_list)
    if is_any_w8a8sc and len(linear_names) > 1:
        if is_anti:
            return PackType.MIX_W8A8SC_ANTI
        else:
            return PackType.MIX_W8A8SC
    elif is_any_w8a8sc and len(linear_names) == 1:
        if is_anti:
            return PackType.ALL_W8A8SC_ANTI
        else:
            return PackType.ALL_W8A8SC
    return PackType.ALL_FP


def is_w8a8(type_desc):
    if type_desc in QUANT_W8A8_DESC_LIST:
        return True
    else:
        return False


def calc_w8a8_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if pack_name:
        quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
        if quant_desc in QUANT_W8A8_DESC_LIST:
            return PackType.ALL_W8A8
        elif quant_desc == QuantType.FLOAT:
            return PackType.ALL_FP

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    norm_anti_desc = f'{norm_name}.module.weight'
    is_anti = True if norm_anti_desc in weights.quant_desc else False
    is_w8a8_list = [is_w8a8(linear_desc) for linear_desc in linear_desces]

    is_all_w8a8 = all(is_w8a8_list)
    is_any_w8a8 = any(is_w8a8_list)

    if is_anti:
        if is_all_w8a8:
            return PackType.ALL_W8A8_ANTI
        elif is_any_w8a8:
            return PackType.MIX_W8A8_ANTI
    else:
        if is_all_w8a8:
            return PackType.ALL_W8A8
        elif is_any_w8a8:
            return PackType.MIX_W8A8
    return PackType.ALL_FP


def calc_w8a16_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if pack_name:
        quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
        if quant_desc == QuantType.W8A16.upper():
            return PackType.ALL_W8A16
        elif quant_desc == QuantType.FLOAT:
            return PackType.ALL_FP
    
    norm_anti_desc = f'{norm_name}.module.weight'

    is_anti = True if norm_anti_desc in weights.quant_desc else False

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    is_w8a16_list = [linear_desc == QuantType.W8A16.upper() for linear_desc in linear_desces]

    is_all_w8a16 = all(is_w8a16_list)
    is_any_w8a16 = any(is_w8a16_list)

    if is_anti:
        if is_all_w8a16:
            return PackType.ALL_W8A16_ANTI
        elif is_any_w8a16:
            return PackType.MIX_W8A16_ANTI
    else:
        if is_all_w8a16:
            return PackType.ALL_W8A16
        elif is_any_w8a16:
            return PackType.MIX_W8A16
    return PackType.ALL_FP


def calc_w4a16_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if pack_name:
        quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
        if quant_desc == QuantType.W4A16.upper():
            return PackType.ALL_W4A16
        elif quant_desc == QuantType.FLOAT:
            return PackType.ALL_FP
    
    norm_anti_desc = f'{norm_name}.module.weight'

    is_anti = True if norm_anti_desc in weights.quant_desc else False

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    is_w4a16_list = [linear_desc == QuantType.W4A16.upper() for linear_desc in linear_desces]

    is_all_w4a16 = all(is_w4a16_list)
    is_any_w4a16 = any(is_w4a16_list)

    if is_anti:
        if is_all_w4a16:
            return PackType.ALL_W4A16_ANTI
        elif is_any_w4a16:
            return PackType.MIX_W4A16_ANTI
    else:
        if is_all_w4a16:
            return PackType.ALL_W4A16
        elif is_any_w4a16:
            return PackType.MIX_W4A16
    return PackType.ALL_FP


def calc_w8a8_dynamic_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if pack_name:
        quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
        if quant_desc == QuantType.W8A8_DYNAMIC.upper():
            return PackType.ALL_W8A8_DYNAMIC
        elif quant_desc == QuantType.FLOAT:
            return PackType.ALL_FP
    
    norm_anti_desc = f'{norm_name}.module.weight'

    is_anti = True if norm_anti_desc in weights.quant_desc else False

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    is_w8a8_dynamic_list = [linear_desc == QuantType.W8A8_DYNAMIC.upper() for linear_desc in linear_desces]

    is_all_w8a8_dynamic = all(is_w8a8_dynamic_list)
    is_any_w8a8_dynamic = any(is_w8a8_dynamic_list)

    if is_anti:
        if is_all_w8a8_dynamic:
            return PackType.ALL_W8A8_DYNAMIC_ANTI
        elif is_any_w8a8_dynamic:
            return PackType.MIX_W8A8_DYNAMIC_ANTI
    else:
        if is_all_w8a8_dynamic:
            return PackType.ALL_W8A8_DYNAMIC
        elif is_any_w8a8_dynamic:
            return PackType.MIX_W8A8_DYNAMIC
    return PackType.ALL_FP


def calc_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if weights.quantize in [QuantType.W8A8, QuantType.W8A8S]:
        pack_type = calc_w8a8_linear_pack_type(weights, linear_names, norm_name, pack_name)
    elif weights.quantize == QuantType.W4A16:
        pack_type = calc_w4a16_linear_pack_type(weights, linear_names, norm_name, pack_name)
    elif weights.quantize == QuantType.W8A16:
        pack_type = calc_w8a16_linear_pack_type(weights, linear_names, norm_name, pack_name)
    elif weights.quantize == QuantType.W8A8SC:
        pack_type = calc_w8a8sc_linear_pack_type(weights, linear_names, norm_name, pack_name)
    elif weights.quantize == QuantType.W8A8_DYNAMIC:
        pack_type = calc_w8a8_dynamic_linear_pack_type(weights, linear_names, norm_name, pack_name)
    else:
        pack_type = PackType.ALL_FP
    return pack_type


class LinearType(int, Enum):
    INVALID = -1
    FP = 0
    INT = 1


class TransposeType(int, Enum):
    INVALID = -1
    NOT_TRANSPOSE = 0
    TRANSPOSE = 1


ALL_PACK_LIST = [
    PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI,
    PackType.ALL_W8A16_ANTI, PackType.ALL_W4A16_ANTI,
    PackType.ALL_W4A16, PackType.ALL_W8A16,
    PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI,
    PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
]

HAS_ANTIOUTLIER = True
HAS_QUANT_ROLLBACK = True


PACK_TYPE_ROUTER = {
    (QuantType.W8A8, not HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.ALL_W8A8,
    (QuantType.W8A8, not HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.ALL_W8A8_ANTI,
    (QuantType.W8A8, HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.MIX_W8A8,
    (QuantType.W8A8, HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.MIX_W8A8_ANTI,
    (QuantType.W8A8S, not HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.ALL_W8A8,
    (QuantType.W8A8S, not HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.ALL_W8A8_ANTI,
    (QuantType.W8A8S, HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.MIX_W8A8,
    (QuantType.W8A8S, HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.MIX_W8A8_ANTI,
    (QuantType.W8A8SC, not HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.ALL_W8A8SC,
    (QuantType.W8A8SC, not HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.ALL_W8A8SC_ANTI,
    (QuantType.W8A8SC, HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.MIX_W8A8SC,
    (QuantType.W8A8SC, HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.MIX_W8A8SC_ANTI,
    (QuantType.W8A8_DYNAMIC, not HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.ALL_W8A8_DYNAMIC,
    (QuantType.W8A8_DYNAMIC, not HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.ALL_W8A8_DYNAMIC_ANTI,
    (QuantType.W8A8_DYNAMIC, HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.MIX_W8A8_DYNAMIC,
    (QuantType.W8A8_DYNAMIC, HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.MIX_W8A8_DYNAMIC_ANTI,
    (QuantType.W8A16, not HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.ALL_W8A16,
    (QuantType.W8A16, not HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.ALL_W8A16_ANTI,
    (QuantType.W8A16, HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.MIX_W8A16,
    (QuantType.W8A16, HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.MIX_W8A16_ANTI,
    (QuantType.W4A16, not HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.ALL_W4A16,
    (QuantType.W4A16, not HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.ALL_W4A16_ANTI,
    (QuantType.W4A16, HAS_QUANT_ROLLBACK, not HAS_ANTIOUTLIER): PackType.MIX_W4A16,
    (QuantType.W4A16, HAS_QUANT_ROLLBACK, HAS_ANTIOUTLIER): PackType.MIX_W4A16_ANTI,
}


def get_pack_type(weights, linear_names, norm_name, pack_name=None):
    if weights.quantize is None or weights.quantize == QuantType.FLOAT:
        return PackType.ALL_FP

    linear_desces = [None]
    if pack_name is not None:
        linear_desces = [weights.quant_desc.get(f'{pack_name}.weight', None)]
    if linear_desces[0] is None:
        linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    unique_linear_desces = set(linear_desces)
    if len(unique_linear_desces) == 1 and list(unique_linear_desces)[0] == QuantType.FLOAT.upper():
        return PackType.ALL_FP
    has_quant_rollback = len(unique_linear_desces) != 1

    if weights.quantize == QuantType.W8A8SC:
        norm_anti_desc = f'{norm_name}.anti.weight'
    else:
        norm_anti_desc = f'{norm_name}.module.weight'
    has_antioutlier = True if norm_anti_desc in weights.quant_desc else False

    return PACK_TYPE_ROUTER.get((weights.quantize, has_quant_rollback, has_antioutlier), PackType.ALL_FP)
