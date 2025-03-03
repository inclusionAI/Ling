# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re
import os
import glob
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from atb_llm.utils.log.logging import logger
from .categories import name_en2zh, subcategories, categories 


choices = ["A", "B", "C", "D"]
category2subject = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                category2subject[k].append(subject)


def extract_choice(response):
    not_in_choices_error = "The answer is not in the list of choices."

    response = str(response)
    if response[0] in choices:
        return response[0]
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])', 1),
        (r'([ABCD]) ?选?项(是|为)?正确', 1),
        (r'正确的?选项(是|为) ?([ABCD])', 2),
        (r'答案(应该)?(是|为)([ABCD])', 3),
        (r'选项 ?([ABCD]) ?(是|为)?正确', 1),
        (r'选择答案 ?([ABCD])', 1),
        (r'答案?：?([ABCD])', 1),
        (r'([ABCD])(选?项)?是?符合题意', 1),
        (r'答案选项：? ?([ABCD])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCD])', 3), # chatgpt

    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            if answer not in choices:
                raise RuntimeError(not_in_choices_error)
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            if answer not in choices:
                raise RuntimeError(not_in_choices_error)
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCD])', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            if answer not in choices:
                raise RuntimeError(not_in_choices_error)
            return answer

    # 4. Check the only mentioend choices
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        if answer not in choices:
            raise RuntimeError(not_in_choices_error)
        return answer

    return choices[random.randint(0, 3)]


def get_results(debug_dir='', result_path=''):
    category_key = "category"
    avg_acc_key = "avg_acc"
    time_costs_key = "time_cost(s)"

    all_acc = defaultdict(float)
    all_time = defaultdict(float)
    all_df = []
    result_stat = {
        category_key: [],
        avg_acc_key: [],
        time_costs_key: []
    }
    for subj in name_en2zh.keys():
        try:
            file = glob.glob(os.path.join(debug_dir, f"results_{subj}.csv"))[0]
        except Exception:
            logger.warning("Warning, %s result file not found", subj)
            continue
        df = pd.read_csv(file, names=['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'response', 'time'], index_col=0)
        # To deal with some mismath between data and answer
        if df.iloc[0]['question'] == '1':
            df = df.drop(0)
        df['pred'] = df['response'].apply(extract_choice)
        df['acc'] = df['answer'] == df['pred']
        acc = np.mean(df['acc']) * 100
        e2e_time = np.mean(df['time'])
        all_acc[subj] = acc
        all_time[subj] = e2e_time
        all_df.append(df)

    all_df = pd.concat(all_df)
    for category, subjects in category2subject.items():
        avg_acc = np.mean(list(map(lambda x: all_acc[x], subjects)))
        avg_time = np.mean(list(map(lambda x: all_time[x], subjects)))
        result_stat[category_key].append(category)
        result_stat[avg_acc_key].append(avg_acc)
        result_stat[time_costs_key].append(avg_time)
        logger.info("%-40s %.2f %.2f", category, avg_acc, avg_time)
    avg_all_acc = np.mean(list(all_acc.values()))
    avg_all_time = np.mean(list(all_time.values()))
    result_stat[category_key].append("Overall")
    result_stat[avg_acc_key].append(avg_all_acc)
    result_stat[time_costs_key].append(avg_all_time)
    logger.info("%-30s %.2f %.2f", 'Overall', avg_all_acc, avg_all_time)
    df = pd.DataFrame(result_stat)
    df.to_csv(result_path, index=False)

    return all_acc