#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
utils
"""
import os
from dataclasses import dataclass

import torch
from atb_llm.utils.env import ENV


FLAG_OS_MAP = {
    'r': os.O_RDONLY, 'r+': os.O_RDWR,
    'w': os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
    'w+': os.O_CREAT | os.O_TRUNC | os.O_RDWR,
    'a': os.O_CREAT | os.O_APPEND | os.O_WRONLY,
    'a+': os.O_CREAT | os.O_APPEND | os.O_RDWR,
    'x': os.O_CREAT | os.O_EXCL,
    "b": getattr(os, "O_BINARY", 0)
}


@dataclass
class TorchParallelInfo:
    __is_initialized: bool = False
    __world_size: int = 1
    __local_rank: int = 0

    def __post_init__(self):
        self.try_to_init()

    @property
    def is_initialized(self):
        return self.__is_initialized

    @property
    def world_size(self):
        _ = self.try_to_init()
        return self.__world_size

    @property
    def local_rank(self):
        _ = self.try_to_init()
        return self.__local_rank

    @property
    def is_rank_0(self) -> bool:
        return self.local_rank == 0

    @staticmethod
    def get_rank() -> int:
        return 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

    @staticmethod
    def get_world_size() -> int:
        return 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()

    def try_to_init(self):
        """
        没有初始化的时候，刷新初始化状态以及world_size local_rank
        :return:
        """
        if not self.__is_initialized:
            is_initialized = torch.distributed.is_initialized()
            if is_initialized:
                self.__local_rank = self.get_rank()
                self.__world_size = self.get_world_size()
            self.__is_initialized = is_initialized
        return self.__is_initialized


def load_atb_speed():
    atb_speed_home_path = ENV.atb_speed_home_path
    lib_path = os.path.join(atb_speed_home_path, "lib", "libatb_speed_torch.so")
    torch.classes.load_library(lib_path)


torch_parallel_info = TorchParallelInfo()
