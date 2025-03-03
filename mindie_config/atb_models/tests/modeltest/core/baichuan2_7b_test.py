# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import logging

from base import model_test
from transformers.generation.utils import GenerationConfig


class Baichuan27BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.tokenizer_params = None

    @staticmethod
    def get_dataset_list():
        return ["BoolQ", "CEval"]
  
    def remove_part_of_generation_config(self, generation_config):
        ori_gen = GenerationConfig()
        diff_dict = generation_config.to_diff_dict()
        logging.info(diff_dict)
        for key in diff_dict:
            if key.endswith("_id"):
                continue
            ori_value = getattr(ori_gen, key, None)
            if ori_value is not None:
                setattr(generation_config, key, getattr(ori_gen, key))
                logging.info("replace %s", key)
        return generation_config

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:2048'
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = '1'
  
    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'revision': None,
            'use_fast': False,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': self.trust_remote_code
        }
    
    def get_supported_model_type(self):
        return ["baichuan"]


def main():
    Baichuan27BModelTest.create_instance()


if __name__ == "__main__":
    main()
