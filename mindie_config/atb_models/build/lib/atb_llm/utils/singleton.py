# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]
