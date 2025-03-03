# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import shutil
import torch
from transformers import LlavaNextForConditionalGeneration

from atb_llm.utils import file_utils
from atb_llm.models.base.model_utils import safe_from_pretrained
from examples.convert.model_slim.quantifier import QuantPipeline, Quantifier, parse_arguments


CPU = "cpu"
NPU = "npu"


MAX_TOKENIZER_FILE_SIZE = 1024 * 1024 * 1024


def need_saved_file(file_name):
    if 'tokenizer' in file_name:
        return file_name
    if 'tokenization' in file_name:
        return file_name
    if 'special_token_map' in file_name:
        return file_name
    if 'added_tokens' in file_name:
        return file_name
    if 'preprocessor_config' in file_name:
        return file_name
    return False


class QuantPipelineLlava(QuantPipeline):
    def __init__(self, context):
        super().__init__(context)

    def get_quantifier(self):
        quant_conf = self.get_quant_conf()
        anti_outlier_conf = self.get_anti_outlier_conf()
        quantifier = QuantifierLlava(
            self.context.model_path, quant_conf, anti_outlier_conf,
            device_type=self.context.device_type, tokenizer_args=json.loads(self.context.tokenizer_args),
            model_name=self.context.model_name,
        )

        return quantifier


class QuantifierLlava(Quantifier):
    def __init__(self, model_path_or_name, quant_config=None,
                 anti_outlier_config=None, device_type='cpu', **kwargs):
        super().__init__(model_path_or_name,
                         quant_config,
                         anti_outlier_config,
                         device_type,
                         **kwargs)
    
    def get_model(self, **kwargs):
       
        device_map = CPU if self.device_type == CPU else "auto"
        dtype = self.config.torch_dtype if self.device_type == NPU else torch.float32
        return safe_from_pretrained(LlavaNextForConditionalGeneration,
                                          self.model_path_or_name,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=dtype,
                                          device_map=device_map,
                                          use_safetensors=True,
                                          trust_remote_code=False)


def copy_process_files(model_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for filename in file_utils.safe_listdir(model_dir):
        if need_saved_file(filename):
            src_filepath = os.path.join(model_dir, filename)
            src_filepath = file_utils.standardize_path(src_filepath)
            file_utils.check_file_safety(src_filepath, 'r', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            dest_filepath = os.path.join(dest_dir, filename)
            dest_filepath = file_utils.standardize_path(dest_filepath)
            file_utils.check_file_safety(dest_filepath, 'w', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            shutil.copyfile(src_filepath, dest_filepath)


if __name__ == "__main__":
    args = parse_arguments()
    QuantPipelineLlava(args).run()
    copy_process_files(args.model_path, args.save_directory)
