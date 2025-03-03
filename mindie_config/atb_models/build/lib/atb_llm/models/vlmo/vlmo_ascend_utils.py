# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import copy
import time
import logging as logger
import json
from transformers import BertTokenizer
import torch
import torch_npu
import numpy as np
from atb_llm.utils.file_utils import safe_open


def label_2_ans(path):
    ans2label_file = os.path.join(path, "answer2label.txt")
    ans2label = {}
    label2ans = []
    with safe_open(ans2label_file, mode="r", encoding="utf-8") as reader:
        for i, line in enumerate(reader):
            data = json.loads(line)
            ans = data["answer"]
            label = data["label"]
            label = int(label)
            ans2label[ans] = i
            label2ans.append(ans)
    return label2ans


def safe_iter(database):
    it = iter(database)
    while True:
        try:
            yield next(it)
        except StopIteration:
            break
        except Exception as e:
            logger.info("wrong key: %s", e)
            break


def get_data(jl1):
    cost1 = np.array([])
    
    count = 0
    right = 0
    wrong = 0
    
    for j1 in jl1:
        count += 1
        if float(j1['cost']) < 1000.0:
            cost1 = np.append(cost1, j1['cost'])
        if j1['res'] in j1['answers']:
            right += 1
        else:
            wrong += 1
            continue
    accuracy = np.divide(right, count)
    logger.info("accuracy %f", accuracy)
    logger.info("count of questions %d", count)
    logger.info("mean of cost: %f ms", np.mean(cost1))     