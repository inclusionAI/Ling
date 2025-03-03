# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import argparse
import os
import shutil
import zipfile
import tarfile
import gzip
import requests
from tqdm import tqdm
from atb_llm.utils.log.logging import logger
from atb_llm.utils.file_utils import safe_open

MAX_FILE_SIZE = 209715200 # 200MB
supported_dataset_list = [
    "boolq",
    "ceval",
    "cmmlu",
    "humaneval",
    "humaneval_x",
    "gsm8k",
    "longbench",
    "mmlu",
    "needlebench",
    "truthfulqa"
]
script_path = os.path.abspath(__file__)
temp_data_folder_path = os.path.join(os.path.dirname(os.path.dirname(script_path)), "temp_data")
data_folder_path = os.path.join(os.path.dirname(os.path.dirname(script_path)), "data")


def setup_parser():
    parser = argparse.ArgumentParser(description="dataset download...")
    parser.add_argument(
        "--dataset_name",
        type=lambda s: [item.strip().lower() for item in s.split(',')],
        help="dataset to be downloaded"
    )
    parser.add_argument(
        "--remove_cache",
        action="store_true",
        help="remove all cache before download"
    )
    parser.add_argument(
        "--remove_temp",
        action="store_true",
        help="remove all temp files after download"
    )
    return parser


def organize_boolq(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
    logger.info(f"Boolq data is ready.")


def organize_longbench(src_path, dest_path):
    with zipfile.ZipFile(os.path.join(src_path, "data.zip"), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    logger.info(f"Longbench data is ready.")


def organize_gsm8k(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
    os.rename(
        os.path.join(dest_path, "test.jsonl"),
        os.path.join(dest_path, "GSM8K.jsonl"))
    logger.info(f"Gsm8k data is ready.")


def organize_needlebench(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
    logger.info(f"Needlebench data is ready.")


def organize_cmmlu(src_path, dest_path):
    with zipfile.ZipFile(os.path.join(src_path, "cmmlu_v1_0_1.zip"), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    logger.info(f"Cmmlu data is ready.")


def organize_ceval(src_path, dest_path):
    with zipfile.ZipFile(os.path.join(src_path, "ceval-exam.zip"), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    logger.info(f"Ceval data is ready.")


def organize_mmlu(src_path, dest_path):
    with tarfile.open(os.path.join(src_path, "data.tar"), "r:") as tar_ref:
        members = tar_ref.getmembers()
        for member in members:
            member.name = os.path.relpath(member.name, start=member.name.split('/')[0])
        tar_ref.extractall(path=dest_path, members=members)
    logger.info(f"Mmlu data is ready.")


def organize_humaneval(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
    os.remove(os.path.join(dest_path, "HumanEval.jsonl.gz"))
    with gzip.open(os.path.join(src_path, "HumanEval.jsonl.gz"), "rb") as f_in:
        with safe_open(os.path.join(dest_path, "HumanEval.jsonl"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logger.info(f"Humaneval data is ready.")


def organize_humaneval_x(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
    logger.info(f"Humaneval-x data is ready.")


def organize_truthfulqa(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
    logger.info(f"Truthfulqa data is ready.")


def organize():
    os.makedirs(data_folder_path, exist_ok=True)
    logger.info(f"Dataset to be organized: {dataset_list}")
    for dataset in tqdm(dataset_list):
        src_path = os.path.join(temp_data_folder_path, dataset)
        dest_path = os.path.join(data_folder_path, dataset)
        shutil.rmtree(dest_path, ignore_errors=True)
        try:
            func = globals()[f"organize_{dataset}"]
            func(src_path, dest_path)
        except KeyError:
            logger.error(f"No function named organize_{dataset} found.")
    if remove_temp:
        shutil.rmtree(temp_data_folder_path, ignore_errors=True)


if __name__ == "__main__":
    data_parser = setup_parser()
    args = data_parser.parse_args()
    dataset_list = args.dataset_name
    remove_cache = args.remove_cache
    remove_temp = args.remove_temp
    if not dataset_list:
        dataset_list = supported_dataset_list
    organize()

