#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
from unittest import TestCase

from atb_speed.common.config import atb_speed_config


class ConfigTest(TestCase):
    def test_1(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        atb_speed_config.init_config(os.path.join(pwd, "template.ini"))
        self.assertEqual(atb_speed_config.performance.batch_size, 1)
