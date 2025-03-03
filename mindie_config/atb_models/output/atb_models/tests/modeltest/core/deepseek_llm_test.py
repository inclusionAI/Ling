# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
import os
import json
from base import model_test



class DeepseekLlmModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[10]
        model_name = "deepseek_llm"
        config_path = os.path.join(weight_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
            prefix_dict = {
                4096: "deepseek_llm_",
            }
            model_ver = {
                30: "7b",
                95: "67b",
            }
            if "max_position_embeddings" in self.config_data and "num_hidden_layers" in self.config_data:
                model_name = prefix_dict.get(self.config_data["max_position_embeddings"])\
                             + model_ver.get(self.config_data["num_hidden_layers"])
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)
    
    @staticmethod
    def get_chip_num():
        return 8

    @staticmethod
    def get_dataset_list():
        return ["BoolQ", "CEval"]
        
    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['INF_NAN_MODE_ENABLE'] = "0"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"
        os.environ['ATB_LAUNCH_KERNAL_WITH_TILING'] = "1"
    
    def get_supported_model_type(self):
        return ["llama"]


def main():
    DeepseekLlmModelTest.create_instance()

if __name__ == "__main__":
    main()