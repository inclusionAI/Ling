# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import argparse
from pathlib import Path
import shutil

from atb_llm.utils.hub import weight_files
from encrypt import EncryptTools
from atb_llm.utils import file_utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_path', type=str, help="input model and tokenizer path")
    parser.add_argument('--encrypted_model_weights_path', type=str, help="output model and tokenizer path")
    parser.add_argument('--key_path', type=str, help="")
    return parser.parse_args()


def encrypt_weights(model_weights_path, encrypted_model_weights_path, key_path):
    file_suffix = ['.bin', '.safetensors']
    try:
        local_weight_files = weight_files(model_weights_path, extension=file_suffix[0])
    except FileNotFoundError:
        local_weight_files = weight_files(model_weights_path, extension=file_suffix[1])
    encrypted_weight_files = [Path(encrypted_model_weights_path) / f"{p.stem.lstrip('pytorch_')}.safetensors"
                              if p.suffix == file_suffix[0] else Path(encrypted_model_weights_path) / f"{p.name}"
                              for p in local_weight_files]

    local_all_files = list(Path(model_weights_path).rglob('*'))
    for local_all_file in local_all_files:
        if file_suffix[0] == local_all_file.suffix or file_suffix[1] == local_all_file.suffix: 
            continue
        output_file = os.path.join(encrypted_model_weights_path, local_all_file.name)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_file = file_utils.standardize_path(str(output_file), check_link=False)
        file_utils.check_file_safety(output_file, 'w', is_check_file_size=False)
        shutil.copyfile(local_all_file, output_file)
    my_encrypt = EncryptTools(key_path)
    my_encrypt.encrypt_files(local_weight_files, encrypted_weight_files, discard_names=[])
    _ = weight_files(encrypted_model_weights_path)


def main():
    args = parse_arguments()
    model_weights_path = file_utils.standardize_path(args.model_weights_path, check_link=False)
    file_utils.check_path_permission(model_weights_path)

    os.makedirs(os.path.dirname(args.encrypted_model_weights_path), exist_ok=True)
    encrypted_model_weights_path = file_utils.standardize_path(args.encrypted_model_weights_path, check_link=False)
    file_utils.check_path_permission(encrypted_model_weights_path)

    os.makedirs(os.path.dirname(args.key_path), exist_ok=True)
    key_path = file_utils.standardize_path(args.key_path, check_link=False)
    file_utils.check_path_permission(key_path)

    encrypt_weights(model_weights_path, encrypted_model_weights_path, key_path)

if __name__ == '__main__':
    main()