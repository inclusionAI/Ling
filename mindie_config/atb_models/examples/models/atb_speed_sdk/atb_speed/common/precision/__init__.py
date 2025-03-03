#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
common launcher
"""
from atb_speed.common.config import atb_speed_config

from .base import CEVALPrecisionTest, MMLUPrecisionTest


def get_precision_test_cls(mode=""):
    """

    :return:
    """
    cls_map = {
        "mmlu": MMLUPrecisionTest,
        "ceval": CEVALPrecisionTest
    }
    return cls_map.get(mode or atb_speed_config.precision.mode.lower(), CEVALPrecisionTest)
