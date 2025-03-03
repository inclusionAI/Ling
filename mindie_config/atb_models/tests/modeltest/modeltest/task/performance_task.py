# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from modeltest.api.task import Task


class PerformanceTask(Task):
    def __init__(self, config_path) -> None:
        super().__init__('performance', config_path)