# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

from atb_llm.utils import file_utils


def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
    dataset = []
    if dataset_path == "./atb_llm/models/qwen2/humaneval_x.jsonl":
        key_name = 'prompt'
    with file_utils.safe_open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file_utils.safe_readlines(file):
            data = json.loads(line)
            text = data.get(key_name, line)
            dataset.append(text)
    return dataset
