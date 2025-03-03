# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import re
import glob
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from modeltest.metric.acc import AccMetric
from atb_llm.utils.log.logging import logger
from .precision_task import PrecisionTask


CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering", "statistics"],
    "Humanities": ["history", "philosophy", "law", "arts", "literature", "global"],
    "Social Science": ['linguistics', "business", "politics", "culture", "economics", 
                    "geography", "psychology", "education", "sociology"],
    "Other":["other"],
    "China specific": ["china specific"]
}


class CMMLUPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.labels = []
        self.batch_num = []
        self.all_preds = []

    def prepare_data(self, metric):
        for _ in self.task_config.subject_mapping:
            if isinstance(metric, AccMetric):
                metric.correct_num_list.append(0)
                self.batch_num.append(0)
                self.all_preds.append([])
        _, val_list, header_list = self.load_dataset_by_task_name()

        for sub_dataset_idx, _ in enumerate(val_list):
            question_list = list(header_list[sub_dataset_idx].Question.values)

            for data_idx in range(len(val_list[sub_dataset_idx])):
                val_list[sub_dataset_idx][data_idx].insert(0, question_list[data_idx])

        return val_list

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        dev_list, val_list, header_list = self.load_dataset_by_task_name()
        dev_data = dev_list[sub_dataset_idx]
        prompt_ends = [self.format_example(batched_data, j, include_answer=False)
                    for j in range(len(batched_data))]
        prompts = [self.gen_prompt(dev_data=dev_data, 
                                  subject=list(self.task_config.subject_mapping.items())[sub_dataset_idx][0],
                                  prompt_end=prompt_end,
                                  num_few_shot=self.task_config.shots,
                                  tokenizer=self.tokenizer)
                                  for prompt_end in prompt_ends]
        self.labels = [batched_data[j][len(batched_data[0]) - 1] 
                      for j in range(len(batched_data))]

        return prompts
    
    def load_dataset_by_task_name(self, format_path=".csv"):
        dev_list, val_list, header_list = [], [], []
        row_begin_idx, col_begin_idx = 1, 1
        for _, subject_name in enumerate(self.task_config.subject_mapping):
            origin_dev_df = pd.read_csv(os.path.join(self.task_config.local_dataset_path, "dev", 
                                subject_name + format_path), header=None, index_col=0)
            origin_val_df = pd.read_csv(os.path.join(self.task_config.local_dataset_path, "test",  
                                subject_name + format_path), header=None, index_col=0)
            header = pd.read_csv(os.path.join(self.task_config.local_dataset_path, "test", 
                                subject_name + format_path))
            dev_df = origin_dev_df.iloc[row_begin_idx:row_begin_idx + self.task_config.shots, col_begin_idx:]
            val_df = origin_val_df.iloc[row_begin_idx:, col_begin_idx:]
            dev_data = dev_df.values.tolist()
            val_data = val_df.values.tolist()
            dev_list.append(dev_data)
            val_list.append(val_data)
            header_list.append(header)
        return dev_list, val_list, header_list
    
    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        choice_ids = [self.tokenizer.convert_tokens_to_ids(choice) 
                     for choice in self.task_config.choices]
        logits = logits[:, choice_ids].detach().cpu().numpy()
        _, val_list, header_list = self.load_dataset_by_task_name()
        self.batch_num[sub_dataset_idx] += len(batched_data)
        for idx, _ in enumerate(batched_data):
            conf = self.softmax(logits[idx])[self.task_config.choices.index(self.labels[idx])]
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}.get(np.argmax(logits[idx]), "Unknown")

            self.all_preds[sub_dataset_idx].append(pred)
            metric.csv_debug.get("golden_result", []).append(self.labels[idx])
            metric.csv_debug.get("test_result", []).append(pred)
            metric.csv_debug.get("pass", []).append(pred == self.labels[idx])

            if pred == self.labels[idx]:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1
        
        if self.batch_num[sub_dataset_idx] == len(val_list[sub_dataset_idx]):
            out_file = os.path.join(metric.debug_dir,
                                f"results_{list(self.task_config.subject_mapping.keys())[sub_dataset_idx]}.csv")
            acc = np.mean(metric.correct_num_list)
            logger.debug("Average accuracy %.3f - %s", acc,
                        list(self.task_config.subject_mapping.keys())[sub_dataset_idx])
            header_list[sub_dataset_idx]['prediction'] = self.all_preds[sub_dataset_idx]
            header_list[sub_dataset_idx]['e2etime'] = 0
            header_list[sub_dataset_idx].to_csv(out_file, header=None)

            if sub_dataset_idx == (len(self.task_config.subject_mapping) - 1):
                self.save_result(metric)

    def format_example(self, data, idx, include_answer=True, cot=False):
        prompt_start = "题目："
        prompt = prompt_start + data[idx][0]
        k = len(data[0]) - 2
        for j in range(k):
            prompt += f"\n{self.task_config.choices[j]}. {data[idx][j + 1]}"
        if cot:
            prompt += "\n逐步分析并给出答案选项。"
        else:
            prompt += "\n答案是："

        if include_answer:
            prompt += f"{data[idx][k + 1]}\n\n"
        return prompt
    
    def gen_prompt(self, dev_data, subject, prompt_end, num_few_shot=0, tokenizer=None):
        max_length = 2048
        cot = False

        subject_name = self.task_config.subject_mapping.get(subject, "未知主题")['name_en2zh']
        prompt = f"以下是关于{subject_name}的单项选择题，请直接给出正确答案的选项。\n\n"
        
        if tokenizer is None:
            for i in range(num_few_shot):
                example = self.format_example(dev_data, i, subject)
                prompt += example
            return prompt + prompt_end

        start_end_token_len = len(tokenizer.encode(prompt) + tokenizer.encode(prompt_end))
        if start_end_token_len > max_length:
            return prompt_end
        
        prompt_list = []
        if num_few_shot > 0:
            for i in range(num_few_shot):
                example = self.format_example(dev_data, i, subject)
                prompt_list.append((example, tokenizer.encode(example)))

            while prompt_list and sum(len(e[1]) for e in prompt_list) >= max_length - start_end_token_len:
                logger.warning("Warning: %d shot case exceeds max_input_length, remove 1 shot.", 
                                    len(prompt_list))
                longest_length = max([len(e[1]) for e in prompt_list])
                prompt_list = [e for e in prompt_list if len(e[1]) != longest_length]
            for p in prompt_list:
                prompt += p[0]

        return prompt + prompt_end

    def softmax(self, x):
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax

    def extract_choice(self, response):
        not_in_choices_error = "The answer is not in the list of choices."

        response = str(response)
        if response[0] in self.task_config.choices:
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
                if answer not in self.task_config.choices:
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
                if answer not in self.task_config.choices:
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
                if answer not in self.task_config.choices:
                    raise RuntimeError(not_in_choices_error)
                return answer

        # 4. Check the only mentioend choices
        pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
        m = re.match(pattern, response)
        if m:
            answer = m.group(1)
            if answer not in self.task_config.choices:
                raise RuntimeError(not_in_choices_error)
            return answer

        return self.task_config.choices[random.randint(0, 3)]


    def get_results(self, debug_dir='', result_path=''):
        category_key = "category"
        avg_acc_key = "avg_acc"
        time_costs_key = "time_cost(s)"
        category2subject = defaultdict(list)
        for k, v in CATEGORIES.items():
            for subject, subcat in self.task_config.subject_mapping.items():
                for c in subcat['subcategories']:
                    if c in v:
                        category2subject[k].append(subject)

        all_acc = defaultdict(float)
        all_time = defaultdict(float)
        all_df = []
        result_stat = {
            category_key: [],
            avg_acc_key: [],
            time_costs_key: []
        }
        for subj in self.task_config.subject_mapping.keys():
            try:
                file = glob.glob(os.path.join(debug_dir, f"results_{subj}.csv"))[0]
            except Exception:
                logger.warning("Warning, %s result file not found", subj)
                continue
            df = pd.read_csv(file, 
                            names=['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'response', 'time'], 
                            index_col=0)
            # To deal with some mismath between data and answer
            if df.iloc[0]['question'] == '1':
                df = df.drop(0)
            df['pred'] = df['response'].apply(self.extract_choice)
            df['acc'] = df['answer'] == df['pred']
            df['time'] = 0
            acc = np.mean(df['acc']) * 100
            all_acc[subj] = acc
            all_time[subj] = 0
            all_df.append(df)

        all_df = pd.concat(all_df)
        for category, subjects in category2subject.items():
            avg_acc = np.mean(list(map(lambda x: all_acc[x], subjects)))
            avg_time = np.mean(list(map(lambda x: all_time[x], subjects)))
            result_stat[category_key].append(category)
            result_stat[avg_acc_key].append(avg_acc)
            result_stat[time_costs_key].append(avg_time)
            logger.info("%-40s %.2f", category, avg_acc)
        avg_all_acc = np.mean(list(all_acc.values()))
        avg_all_time = np.mean(list(all_time.values()))
        result_stat[category_key].append("Overall")
        result_stat[avg_acc_key].append(avg_all_acc)
        result_stat[time_costs_key].append(avg_all_time)
        logger.info("%-30s %.2f", 'Overall', avg_all_acc)
        df = pd.DataFrame(result_stat)
        df.to_csv(result_path, index=False)

        return all_acc

    def save_result(self, metric):
        debug_dir = metric.debug_dir
        result_dir = metric.result_dir
        result_dir = os.path.join(result_dir, "CMMLU_result.csv")
        try:
            self.get_results(debug_dir, result_dir)
        except Exception as e:
            logger.error("Please check the debug path and result path")
        logger.info("CMMLU result saved to: %s", result_dir)
        