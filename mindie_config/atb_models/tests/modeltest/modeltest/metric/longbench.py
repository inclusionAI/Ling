# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import math
import numpy as np
from tabulate import tabulate
from atb_llm.utils.file_utils import safe_open
from modeltest.api.metric import Metric
from atb_llm.utils.log.logging import logger


class LongbenchMetric(Metric):
    def __init__(self, *args) -> None:
        super().__init__("longbench", *args)
        self.update_times = 0
        self.score_map = {"0-4k": [], "4-8k": [], "8k+": []}
        self.score_map_stat = {"0-4k": [], "4-8k": [], "8k+": []}
        self.task_scores_map = {}
        self.scores = 0.
        self.scores_stat = 0.
        self.task_scores = {}
        self.final_score_list = []
        self.final_score = 0
        self.struct_map = {
            "longbench": self.task_scores,
            "longbench_e": self.task_scores_map
        }
        self.csv_debug = {
            "key": [],
            "queries": [],
            "input_token_ids": [],
            "output_token_ids": [], 
            "test_result": [],
            "golden_result": [],
            "len": [],
            "all_class": []
        }
    
    def print_metric(self):
        self.__process_final_scores()
        data = [
            [
                self.task_config.task_name,
                self.task_config.shots,
                self.case_num,
                "longbench",
                self.final_score
            ]
        ]
        case_num_iter = iter(self.case_num_list)
        final_score_iter = iter(self.final_score_list)
        for _, desc in self.task_config.subject_mapping.items():
            if desc["type"] == self.task_config.task_name:
                data.append([desc["name"],
                            self.task_config.shots,
                            next(case_num_iter),
                            "longbench",
                            next(final_score_iter)])
        headers = ["Tasks", "N-Shot", "Total", "Metric", "Value"]
        markdown_table = tabulate(data, headers, tablefmt="grid")
        logger.info(f"Metric Table:\n{markdown_table}")
        with safe_open(self.result_file_path, mode='w', encoding='utf-8') as file:
            file.write(f"runner config:\n{self.runner_config}\n")
            file.write(f"model config:\n{self.model_config}\n")
            file.write(f"task config:\n{self.task_config}\n")
            file.write(f"metric table:\n{markdown_table}\n")
        super().print_metric()
    
    def __process_final_scores(self):
        for _, res in self.struct_map.get(self.task_config.task_name).items():
            if isinstance(res, dict):
                task_score_list = []
                for _, score in res.items():
                    if not math.isnan(score):
                        task_score_list.append(score)
                task_scores = round(np.mean(task_score_list), 2)
            else:
                task_scores = res
            self.final_score_list.append(task_scores)
        self.final_score = round(np.average(self.final_score_list, weights=self.case_num_list), 2)
