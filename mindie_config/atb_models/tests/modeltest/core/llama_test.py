# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
from base import model_test


class LlamaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[10]
        model_name = "llama"
        config_path = os.path.join(weight_dir, "config.json")
        num_hidden_layers = "num_hidden_layers"
        max_position_embeddings = "max_position_embeddings"
        tie_word_embeddings = "tie_word_embeddings"

        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
            is_llama1 = True if "num_key_value_heads" not in self.config_data else False
            is_llama3 = self.config_data["vocab_size"] == 128256
            prefix_dict = {
                4096: "llama2_",
                8192: "llama3_",
                131072: "llama3.1_" if not self.config_data[tie_word_embeddings] else "llama3.2_",
            } 
            model_ver = {
                16: "1b",
                28: "3b",
                32: "8b" if is_llama3 else "7b",
                40: "13b",
                60: "33b",
                80: "65b" if is_llama1 else "70b",
                96: "175b",
                126: "405b",
            }
            if is_llama1 and num_hidden_layers in self.config_data:
                model_name = "llama_" + model_ver.get(self.config_data[num_hidden_layers], None)
            elif max_position_embeddings in self.config_data and num_hidden_layers in self.config_data:
                model_name = (
                prefix_dict.get(self.config_data[max_position_embeddings], None) + 
                model_ver.get(self.config_data[num_hidden_layers], None)
                )       

        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)
        self.tokenizer_params = None

    @staticmethod  
    def get_chip_num():
        return 8
        
    @staticmethod
    def get_dataset_list():
        return ["BoolQ", "CEval", "GSM8K", "MMLU"]

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
    LlamaModelTest.create_instance()

if __name__ == "__main__":
    main()