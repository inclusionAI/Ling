# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from modeltest.api.task import Task


class PrecisionTask(Task):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.set_environ()
    
    @staticmethod
    def get_batched_data(data, batch_size):
        """
        将数据集分割成多个批次。

        :param data: 数据集，一个包含字典的列表
        :param batch_size: 每个批次的大小
        :return: 生成器，每次迭代返回一个批次的数据
        """
        batch = []
        for idx, item in enumerate(data):
            batch.append(item)
            if len(batch) == batch_size:
                yield idx, batch
                batch = []
        if batch:  # 如果最后一个批次不足batch_size，也返回
            yield idx, batch
    
    def set_environ(self):
        os.environ["LCCL_DETERMINISTIC"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "0"
        os.environ["MODELTEST_DATASET_SPECIFIED"] = self.task_config.task_name

    def inference(self):
        pass