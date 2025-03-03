# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
from base import model_test

QWEN_TEST_NUM_HIDDEN_LAYERS = "num_hidden_layers"
QWEN_TEST_INTERMEDIATE_SIZE = "intermediate_size"
QWEN_TEST_MAX_WINDOW_LAYERS = "max_window_layers"
QWEN_TRANSFORMERS_VERSION = "transformers_version"


class QwenModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        self.tokenizer_params = None
        weight_dir = args[10]
        model_name = "qwen"
        config_path = os.path.join(weight_dir, "config.json")
        model_type = "model_type"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            if config_data[model_type] == "qwen":
                if QWEN_TEST_NUM_HIDDEN_LAYERS in config_data:
                    if config_data[QWEN_TEST_NUM_HIDDEN_LAYERS] == 40:
                        model_name = "qwen_14b"
                    elif config_data[QWEN_TEST_NUM_HIDDEN_LAYERS] == 80:
                        model_name = "qwen_72b"
                    elif config_data[QWEN_TEST_NUM_HIDDEN_LAYERS] == 32:
                        model_name = "qwen_7b"
            elif config_data[model_type] == "qwen2":
                if (QWEN_TEST_INTERMEDIATE_SIZE in config_data) and (QWEN_TEST_MAX_WINDOW_LAYERS in config_data):
                    if config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 2816:
                        model_name = "qwen1.5_0.5b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 5504:
                        model_name = "qwen1.5_1.8b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 6912:
                        model_name = "qwen1.5_4b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 11008:
                        model_name = "qwen1.5_7b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 13696:
                        model_name = "qwen1.5_14b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 27392:
                        model_name = "qwen1.5_32b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 27648:
                        model_name = "qwen2.5_32b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 24576:
                        model_name = "qwen1.5_72b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 18944:
                        if config_data[QWEN_TRANSFORMERS_VERSION] == "4.41.2":
                            model_name = "qwen2_7b"
                        elif config_data[QWEN_TRANSFORMERS_VERSION] == "4.43.1":
                            model_name = "qwen2.5_7b"
                        elif config_data[QWEN_TRANSFORMERS_VERSION] == "4.44.0":
                            model_name = "qwen2.5_coder_7b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 29568:
                        if config_data[QWEN_TEST_MAX_WINDOW_LAYERS] == 80:
                            model_name = "qwen2_72b"
                        elif config_data[QWEN_TEST_MAX_WINDOW_LAYERS] == 70:
                            model_name = "qwen2.5_72b"
                    elif config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 13824:
                        model_name = "qwen2.5_14b"
            elif config_data[model_type] == "qwen2_moe":
                if QWEN_TEST_INTERMEDIATE_SIZE in config_data:
                    if config_data[QWEN_TEST_INTERMEDIATE_SIZE] == 5632:
                        model_name = "qwen2_moe"

        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)

    @staticmethod
    def get_dataset_list():
        return ["CEval", "BoolQ", "GSM8K", "MMLU"]

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = '1'

    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'pad_token': '<|extra_0|>',
            'eos_token': '<|endoftext|>',
            'padding_side': 'left',
            'trust_remote_code': self.trust_remote_code
        }

    def get_supported_model_type(self):
        return ["qwen", "qwen2", "qwen2_moe"]


def main():
    QwenModelTest.create_instance()


if __name__ == "__main__":
    main()
