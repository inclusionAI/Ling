#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
"""
decorator
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps, partial
from typing import List
from typing import Union
from atb_llm.utils.env import ENV


@dataclass
class TimeData:
    step: int = 0
    time_cost: Union[float, int] = 0


@dataclass
class SeqTimeData:
    task_id: str = ""
    time_data_list: List[TimeData] = field(default_factory=list)

    @property
    def generated_tokens(self):
        return len(self.time_data_list)

    @property
    def first_token_delay(self):
        return self.time_data_list[0].time_cost if self.time_data_list else 0

    @property
    def next_token_avg_delay(self):
        if self.generated_tokens <= 1:
            return 0
        return sum(item.time_cost for item in self.time_data_list[1:]) / (self.generated_tokens - 1)


class Timer:
    """
    CommonDecorator
    """
    step: int = 0
    timeit_res: SeqTimeData = SeqTimeData(str(uuid.uuid4()))

    @classmethod
    def reset(cls):
        cls.step = 0
        cls.timeit_res = SeqTimeData(str(uuid.uuid4()))

    @classmethod
    def sync(cls):
        ...

    @classmethod
    def timing(cls, func=None, *, logger=None, level=logging.INFO):
        """
        函数计时
        :return:
        """
        if logger is None:
            logger = logging.getLogger()
        if func is None:
            # 没有括号的时候args是func，有括号的时候args是None
            return partial(Timer.timing, logger=logger, level=level)

        run = cls._timeit_run if ENV.time_it else cls._run

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            wrapper
            :param args:
            :param kwargs:
            :return:
            """
            res = run(func, *args, **kwargs)
            return res

        return wrapper

    @classmethod
    def _run(cls, func, *args, **kwargs):
        res = func(*args, **kwargs)
        return res

    @classmethod
    def _timeit_run(cls, func, *args, **kwargs):
        cls.sync()
        start_time = time.time()
        res = func(*args, **kwargs)
        cls.sync()
        end_time = (time.time() - start_time) * 1000  # ms
        cls.timeit_res.time_data_list.append(TimeData(cls.step, end_time))
        cls.step = cls.step + 1
        return res
