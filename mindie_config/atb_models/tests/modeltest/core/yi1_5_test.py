# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os

from base import model_test


class Yi15ModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[10]
        model_name = "yi1.5"
        self.tokenizer_params = None
        config_path = os.path.join(weight_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
            model_ver = {
                32: "_6b",
                48: "_9b",
                60: "_34b",
            }
            if "max_position_embeddings" in self.config_data and "num_hidden_layers" in self.config_data:
                model_name = model_name + model_ver.get(self.config_data["num_hidden_layers"])
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
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
    
    def set_fa_tokenizer_params(self):
        use_fast = True
        if self.config_data["num_hidden_layers"] in [60]:
            use_fast = False
        self.tokenizer_params = {
            'revision': None,
            'use_fast': use_fast,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': self.trust_remote_code
        }
    
    def get_supported_model_type(self):
        return ["llama"]


def main():
    Yi15ModelTest.create_instance()

if __name__ == "__main__":
    main()