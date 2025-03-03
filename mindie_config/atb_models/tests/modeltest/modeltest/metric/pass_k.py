# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import numpy as np
from tabulate import tabulate
from atb_llm.utils.file_utils import safe_open
from modeltest.api.metric import Metric
from atb_llm.utils.log.logging import logger


class PassKMetric(Metric):
    def __init__(self, *args) -> None:
        super().__init__("pass_k", *args)
        self.correct_num = 0
        self.correct_num_list = []
        self.passed_all = []
        self.k = self.task_config.metric.get("k", 1)
        self.csv_debug = {
            "key": [],
            "queries": [],
            "input_token_ids": [],
            "output_token_ids": [], 
            "test_result": [],
            "golden_result": [],
            "pass": [],
            "test_cases": [],
            "cleaned_up_results": []
        }

    def get_k_value(self):
        return self.k

    def evaluate_and_update_metric(self, pass_at_k, passed_all, sub_dataset_idx):
        self.passed_all.extend(passed_all)
        correct_count = sum(passed_all)
        self.correct_num_list[sub_dataset_idx] += correct_count
        self.correct_num += correct_count
        for k, v in pass_at_k.items():
            logger.info(f"Pass@{k}: {v}")
            self.csv_debug[f'pass@{k}'] = v

    def calculate_pass_k(self, total_cases, correct_cases):
        k = self.get_k_value()
        pass_at_k = {f'pass@{int(k)}': self._calculate_pass_at_k_metric(total_cases, correct_cases, k)}
        return pass_at_k

    def print_metric(self):
        logger.info(f"Printing Pass@{self.get_k_value()} metric for {self.task_config.task_name}")
        logger.info(f"case_num_list: {self.case_num_list}")
        logger.info(f"correct_num_list: {self.correct_num_list}")
        logger.info(f"Total cases: {self.case_num}, Correct cases: {self.correct_num}")
        try:
            k_value = self.get_k_value()
            data = [
                [
                    self.task_config.task_name,
                    self.task_config.shots,
                    self.case_num,
                    f"Pass@{k_value}",
                    self.correct_num / self.case_num
                ]
            ]
            if self.task_config.humaneval_x_datasets_selector: # Humaneval_X
                for idx, lang in enumerate(self.task_config.humaneval_x_datasets_selector):
                    logger.info(f"Processing dataset {lang} at index {idx}")
                    data.append([
                        lang,
                        self.task_config.shots,
                        self.case_num_list[idx],
                        f"Pass@{k_value}",
                        self.correct_num_list[idx] / self.case_num_list[idx]
                    ])
            else: # Humaneval
                for idx, sub_dataset in enumerate(self.task_config.subject_mapping.keys()):
                    logger.info(f"Processing dataset {sub_dataset} at index {idx}")
                    data.append([
                        self.task_config.subject_mapping[sub_dataset]["name"],
                        self.task_config.shots,
                        self.case_num_list[idx],
                        f"Pass@{k_value}",
                        self.correct_num_list[idx] / self.case_num_list[idx]
                    ])
            headers = ["Tasks", "N-Shot", "Total", "Metric", "Value"]
            markdown_table = tabulate(data, headers, tablefmt="grid")
            logger.info(f"Metric Table:\n{markdown_table}")
            with safe_open(self.result_file_path, mode='w', encoding='utf-8') as file:
                file.write(f"runner config:\n{self.runner_config}\n")
                file.write(f"model config:\n{self.model_config}\n")
                file.write(f"task config:\n{self.task_config}\n")
                file.write(f"metric table:\n{markdown_table}\n")
            super().print_metric()
        except IndexError as e:
            logger.error(f"IndexError occurred: {e}")
            logger.error(
                "Length of case_num_list: %d, Length of correct_num_list: %d",
                len(self.case_num_list),
                len(self.correct_num_list)
            )
            raise e
        
    def save_debug(self):
        min_len = min(len(v) for v in self.csv_debug.values())
        for key in self.csv_debug:
            if len(self.csv_debug[key]) > min_len:
                self.csv_debug[key] = self.csv_debug[key][:min_len]
            elif len(self.csv_debug[key]) < min_len:
                self.csv_debug[key].extend([None] * (min_len - len(self.csv_debug[key])))        
        super().save_debug()
   
    def _calculate_pass_at_k_metric(self, total_cases, correct_cases, k):
        return 1.0 - np.prod(1.0 - k / np.arange(total_cases - correct_cases + 1, total_cases + 1))