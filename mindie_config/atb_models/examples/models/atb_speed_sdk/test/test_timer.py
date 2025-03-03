#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
@Time   :  2024/2/9 14:46
"""
import logging
from unittest import TestCase

import torch
import torch.nn as nn
from atb_speed.common.timer import Timer

logging.basicConfig(level=logging.NOTSET)


class AddNet(nn.Module):
    def __init__(self, in_dim, h_dim=5, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)

    @Timer.timing
    def forward(self, x_tensor, y_tensor):
        out = torch.cat([x_tensor, y_tensor], dim=1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class TimerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        Timer.reset()
        # Timer.sync= xxxx
        cls.add_net = AddNet(in_dim=2)

    def test_1(self):
        for _ in range(5):
            x_tensor = torch.randn(1, 1)
            y_tensor = torch.randn(1, 1)
            result = self.add_net.forward(x_tensor, y_tensor)
            logging.info(result)
        logging.info(Timer.timeit_res)
        logging.info(Timer.timeit_res.first_token_delay)
        logging.info(Timer.timeit_res.next_token_avg_delay)
