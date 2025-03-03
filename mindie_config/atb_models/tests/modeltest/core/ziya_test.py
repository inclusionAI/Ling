# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from base import model_test


class ZiyaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        model_name = "ziya"
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)

    @staticmethod    
    def get_chip_num():
        return 8
    
    @staticmethod
    def get_dataset_list():
        return ["BoolQ", "HumanEval"]

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['INF_NAN_MODE_ENABLE'] = "0"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "0"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['ATB_LLM_ENABLE_AUTO_TRANSPOSE'] = "0"

    def get_supported_model_type(self):
        return ["llama", "ziya"]


def main():
    ZiyaModelTest.create_instance()

if __name__ == "__main__":
    main()