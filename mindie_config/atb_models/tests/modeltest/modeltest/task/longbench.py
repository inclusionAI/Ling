# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Copyright (c) 2023 THU-KEG & Zhipu AI
import json
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
import jieba
import numpy as np
from fuzzywuzzy import fuzz
from rouge import Rouge
from tqdm import tqdm
from modeltest.metric.longbench import LongbenchMetric
from atb_llm.utils.file_utils import safe_open
from .precision_task import PrecisionTask


@dataclass 
class LongbenchResMeta:
    predictions: list
    golden_answers: list
    lengths: list
    all_classes: list


class LongBenchPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.sub_dataset_list = self.__filter_sub_dataset()
        self.curr_sub_dataset = {}
        self.score_func_map = {
            "longbench": self.__scorer,
            "longbench_e": self.__scorer_e,
        }
        self.calc_score_func_map = {
            "longbench": self.__calc_scores_stats,
            "longbench_e": self.__calc_score_map_stats,
        }
    
    @staticmethod
    def get_npu_runner_extra_args():
        return {"skip_special_tokens": True}
    
    def prepare_data(self, metric):
        max_dataset_file_size = 30 * 1024 * 1024 # max_dataset_file_size for longbench
        longbench_datasets = []
        for sub_dataset_name in tqdm(self.sub_dataset_list):
            entry = os.path.join(
                self.local_dataset_folder,
                self.task_config.local_dataset_path,
                sub_dataset_name)
            dataset = []
            with safe_open(entry, encoding='utf-8', max_file_size=max_dataset_file_size) as f:
                for line in f:
                    line_json = json.loads(line)
                    dataset.append(line_json)
            longbench_datasets.append(dataset)
        return longbench_datasets

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        self.curr_sub_dataset = self.task_config.subject_mapping[self.sub_dataset_list[sub_dataset_idx]]
        self.task_config.requested_max_output_length = self.curr_sub_dataset.get("maxlen")
        prompts_pattern = self.curr_sub_dataset.get("prompt")
        prompts = [prompts_pattern.format(**data) for data in batched_data]
        prompts_truncated = self.__truncate_input(prompts)
        return prompts_truncated
    
    def result_judge(self,
                     metric: LongbenchMetric,
                     generate_token_lists,
                     logits,
                     sub_dataset_idx,
                     batched_data):
        task_result = [{"pred": generate_token_list,
                        "golden": data["answers"], 
                        "all_class": data["all_classes"],
                        "len": data["length"]}
                        for data, generate_token_list in zip(batched_data, generate_token_lists)]
        self.__get_scores(metric, sub_dataset_idx, task_result)
        if metric.update_times == metric.case_num_list[-1]:
            self.calc_score_func_map.get(self.task_config.task_name)(metric)
            metric.update_times = 0
    
        for idx, generate_token_list in enumerate(generate_token_lists):
            metric.csv_debug.get("golden_result", []).append(batched_data[idx]['answers'])
            metric.csv_debug.get("test_result", []).append(generate_token_list)
            metric.csv_debug.get("len", []).append(batched_data[idx]['length'])
            metric.csv_debug.get("all_class", []).append(batched_data[idx]['all_classes'])
    
    def __filter_sub_dataset(self):
        sub_dataset_list = []
        for sub_dataset_name, desc in self.task_config.subject_mapping.items():
            if desc["type"] == self.task_config.task_name:
                sub_dataset_list.append(sub_dataset_name)
        return sub_dataset_list
    
    def __truncate_input(self, prompts):
        truncated_prompts = prompts
        if self.task_config.need_truncate_input:
            max_input_length = self.task_config.requested_max_input_length
            for idx, prompt in enumerate(prompts):
                tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                if len(tokenized_prompt) > max_input_length:
                    half = int(max_input_length / 2)
                    truncated_prompt = (self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                                self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))
                    truncated_prompts[idx] = truncated_prompt
        return truncated_prompts

    def __get_scores(self, metric: LongbenchMetric, sub_dataset_idx, task_result):
        predictions, golden_answers, lengths, all_classes = zip(*[
            (
                item.get("pred"),
                item.get("golden"),
                item.get("len"),
                item.get("all_class")
            )
            for item in task_result
        ])
        longbench_res = LongbenchResMeta(
            predictions=predictions,
            golden_answers=golden_answers,
            lengths=lengths,
            all_classes=all_classes
        )
        self.score_func_map.get(self.task_config.task_name)(
            metric,
            self.sub_dataset_list[sub_dataset_idx],
            longbench_res
        )
        metric.update_times += len(predictions)
    
    def __get_function(self, func_name):
        function = globals().get(func_name)
        if callable(function):
            return function
        else:
            raise RuntimeError(f"No function named '{func_name}' found.")
    
    def __calc_score_map_stats(self, metric: LongbenchMetric):
        for key in metric.score_map.keys():
            metric.score_map_stat[key] = round(100 * np.mean(metric.score_map[key]), 2)
        metric.task_scores_map[self.curr_sub_dataset.get("name")] = metric.score_map_stat
        metric.score_map = {"0-4k": [], "4-8k": [], "8k+": []}

    def __calc_scores_stats(self, metric: LongbenchMetric):
        if metric.case_num_list[-1] != 0:
            metric.scores_stat = round(100 * metric.scores / metric.case_num_list[-1], 2)
        metric.task_scores[self.curr_sub_dataset.get("name")] = metric.scores_stat
        metric.scores = 0.
        
    def __scorer_e(self, metric: LongbenchMetric, dataset, longbench_res: LongbenchResMeta):
        for (prediction, ground_truths, length, all_classes) in zip(longbench_res.predictions, 
                                                                    longbench_res.golden_answers,
                                                                    longbench_res.lengths,
                                                                    longbench_res.all_classes):
            score = 0.
            if dataset.split(".")[0] in ["trec", "trec_e", "triviaqa", "triviaqa_e", "samsum", "samsum_e", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, self.__get_function(self.task_config.subject_mapping[dataset]["metric"])(
                    prediction, 
                    ground_truth, 
                    all_classes=all_classes
                ))
            if length < 4000:
                metric.score_map["0-4k"].append(score)
            elif length < 8000:
                metric.score_map["4-8k"].append(score)
            else:
                metric.score_map["8k+"].append(score)

    def __scorer(self, metric:LongbenchMetric, dataset, longbench_res: LongbenchResMeta):
        for (prediction, ground_truths, all_classes) in zip(longbench_res.predictions, 
                                                            longbench_res.golden_answers,
                                                            longbench_res.all_classes):
            score = 0.
            if dataset.split(".")[0] in ["trec", "trec_e", "triviaqa", "triviaqa_e", "samsum", "samsum_e", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, self.__get_function(self.task_config.subject_mapping[dataset]["metric"])(
                    prediction, 
                    ground_truth, 
                    all_classes=all_classes
                ))
            metric.scores += score


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs.get("all_classes", [])
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list and len(em_match_list) != 0:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score
    

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except ValueError:
        return 0.0
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    if len(prediction) != 0 and len(ground_truth) != 0:
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)