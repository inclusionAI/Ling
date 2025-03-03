#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
common launcher
"""
from atb_speed.common.launcher.base import get_device, DeviceType

if get_device() == DeviceType.npu:
    from atb_speed.common.launcher.npu import Launcher, ParallelLauncher
else:
    from atb_speed.common.launcher.gpu import Launcher, ParallelLauncher
