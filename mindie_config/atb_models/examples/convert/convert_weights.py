# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse

from atb_llm.utils.convert import convert_files
from atb_llm.utils.hub import weight_files
from atb_llm.utils.log import logger
from atb_llm.utils import file_utils
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    return parser.parse_args()


def convert_bin2st(model_path):
    local_pt_files = weight_files(model_path, extension=".bin")
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors"
        for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])
    _ = weight_files(model_path)


def convert_bin2st_from_pretrained(model_path):
    model = safe_get_model_from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype="auto"
    )
    model.save_pretrained(model_path, safe_serialization=True)


if __name__ == '__main__':
    args = parse_arguments()

    input_model_path = file_utils.standardize_path(args.model_path, check_link=False)
    file_utils.check_path_permission(input_model_path)
    try:
        convert_bin2st(input_model_path)
    except RuntimeError:
        logger.warning('convert weights failed with torch.load method, need model loaded to convert')
        convert_bin2st_from_pretrained(input_model_path)