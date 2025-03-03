# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import csv
import os
import argparse
import string
import editdistance
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log import logger


def get_answer(result_path):
    npu_answers = []
    with safe_open(result_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for _, row in enumerate(reader):
            npu_answers.append(row)
    return npu_answers


def compare_strings(str1, str2):
    diff_num = editdistance.eval(str1, str2)
    total_num = len(str1) + len(str2)
    return diff_num / total_num


def run_compare(first_answer_path, second_answer_path):
    npu_answers = get_answer(first_answer_path)
    gpu_answers = get_answer(second_answer_path)
    diff_total = 0
    num = 0

    for idx, gpu_ans in enumerate(gpu_answers):
        if idx == 0:
            continue
        npu_ans = npu_answers[idx]
        npu_file1 = os.path.basename(npu_ans[0])
        npu_file2 = os.path.basename(npu_ans[1])
        gpu_file1 = os.path.basename(gpu_ans[0])
        gpu_file2 = os.path.basename(gpu_ans[1])
        if npu_file1 == gpu_file1 and npu_file2 == gpu_file2:
            npu_response = npu_ans[-1].replace('<|im_end|>', '')
            npu_response = npu_response.translate(str.maketrans('', '', string.punctuation)).lower()
            gpu_response = gpu_ans[-1].translate(str.maketrans('', '', string.punctuation)).lower()
            diff_ratio = compare_strings(npu_response, gpu_response)
            diff_total += diff_ratio
            num += 1

    logger.info(f"Accuracy Gap: {diff_total/num}")


def parse_args():
    parser = argparse.ArgumentParser(description="acc")
    parser.add_argument("--first_answer_path", required=True, type=str, help="first_answer_path.")
    parser.add_argument("--second_answer_path", required=True, type=str, help="second_answer_path.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_compare(args.first_answer_path, args.second_answer_path)

if __name__ == "__main__":
    main()
