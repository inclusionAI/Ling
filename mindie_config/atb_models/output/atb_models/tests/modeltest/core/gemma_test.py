# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
from base import model_test


class GemmaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[10]
        model_name = "gemma"
        config_path = os.path.join(weight_dir, "config.json")
        num_hidden_layers = "num_hidden_layers"
        
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
            if num_hidden_layers in self.config_data:
                if self.config_data[num_hidden_layers] == 18:
                    model_name = "gemma-2b"
                elif self.config_data[num_hidden_layers] == 28:
                    model_name = "gemma-7b"
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)
        self.tokenizer_params = None
    
    @staticmethod
    def get_chip_num():
        return 8

    @staticmethod
    def get_dataset_list():
        return ["BoolQ"]
    
    def get_block_size(self):
        return 64
    
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
        self.tokenizer_params = {
            'revision': None,
            'use_fast': use_fast,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': self.trust_remote_code
        }

    def get_supported_model_type(self):
        return ["gemma"]


def main():
    GemmaModelTest.create_instance()

if __name__ == "__main__":
    main()