# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

from base import model_test


class Codeshell7BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    @staticmethod
    def get_dataset_list():
        return ["HumanEval"]

    def prepare_environ(self):
        # memory
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        # performance
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"

    def get_supported_model_type(self):
        return ["codeshell"]


def main():
    Codeshell7BModelTest.create_instance()

if __name__ == "__main__":
    main()
