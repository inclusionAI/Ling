# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
from base import model_test


class TelechatModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[10]
        self.model_name = "telechat"
        config_path = os.path.join(weight_dir, "config.json")
        n_layer = "n_layer"
        
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
            if n_layer in self.config_data:
                if self.config_data[n_layer] == 30:
                    self.model_name = "telechat-7b"
                elif self.config_data[n_layer] == 38:
                    self.model_name = "telechat-12b"
        updated_args = args[:3] + (self.model_name,) + args[4:]
        super().__init__(*updated_args)

    @staticmethod
    def get_chip_num():
        return 8
    
    @staticmethod
    def get_dataset_list():
        return ["CEval", "BoolQ", "MMLU", "TruthfulQA"]

    def get_block_size(self):
        block_size = 128
        if self.model_name == 'telechat-12b':
            block_size = 96
        return block_size
    
    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['INF_NAN_MODE_ENABLE'] = "0"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"
    
    def get_supported_model_type(self):
        return ["telechat"]


def main():
    TelechatModelTest.create_instance()

if __name__ == "__main__":
    main()