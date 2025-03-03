# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import argparse
import logging
import librosa
from atb_llm.utils.file_utils import safe_open


def renew_librosa(modify_librosa, restore_librosa):
    required_version = "0.10.2.post1"
    current_version = librosa.__version__

    if current_version != required_version:
        raise RuntimeError(f"需要安装 {required_version}, 请安装正确的版本。")
    librosa_path = os.path.dirname(librosa.__file__)
    file_to_modify = os.path.join(librosa_path, "core", 'audio.py')

    with safe_open(file_to_modify, "r") as file:
        lines = file.readlines()
    target_line_number = 170
    target_line = lines[target_line_number - 1]
    if modify_librosa:
        target_line = target_line.replace("if isinstance", "if 0 and isinstance")
    if restore_librosa:
        target_line = target_line.replace("if 0 and isinstance", "if isinstance")
    lines[target_line_number - 1] = target_line
    with safe_open(file_to_modify, "w") as file:
        file.writelines(lines)
    logging.info(f"修改完成：{file_to_modify}")


def parse_args():
    parser = argparse.ArgumentParser(description="librosa_file")
    parser.add_argument("--modify_librosa",
                        default=False,
                        type=bool,
                        help="modify_librosa.")
    parser.add_argument("--restore_librosa",
                        default=False,
                        type=bool,
                        help="restore_librosa.")
    return parser.parse_args()


def main():
    args = parse_args()
    modify_librosa = args.modify_librosa
    restore_librosa = args.restore_librosa
    if modify_librosa == restore_librosa:
        raise RuntimeError("请检查 modify_librosa 和 restore_librosa参数, 设置其中一个为 True")
    renew_librosa(modify_librosa, restore_librosa)


if __name__ == "__main__":
    main()

