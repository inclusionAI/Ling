# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os

from atb_llm.utils.file_utils import safe_open
from base import model_test


class Phi3ModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        model_name = "phi3"
        updated_args = args[:3] + (model_name,) + args[4:]
        weight_dir = args[10]
        config_path = os.path.join(weight_dir, "config.json")
        with safe_open(config_path, 'r') as f:
            self.config_data = json.load(f)
        if "max_position_embeddings" in self.config_data:
            updated_args = updated_args[:12] + \
                        (self.config_data["max_position_embeddings"],) + \
                        updated_args[13:]
        
        super().__init__(*updated_args)
    
    @staticmethod
    def get_dataset_list():
        return ["BoolQ"]

    def get_supported_model_type(self):
        return ["phi3"]

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


def main():
    Phi3ModelTest.create_instance()

if __name__ == "__main__":
    main()