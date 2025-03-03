# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from tabulate import tabulate
from atb_llm.utils.file_utils import safe_open
from modeltest.api.metric import Metric
from atb_llm.utils.log.logging import logger

MC1 = 'MC1'
MC2 = 'MC2'
BLUE_DIFF = "Bleu_diff"
ROUGE1_DIFF = "Rouge1_diff"


class TruthfulqaMetric(Metric):
    def __init__(self, *args) -> None:
        super().__init__("acc", *args)
        self.csv_debug = {
            "key": [],
            "queries": [],
            "input_token_ids": [],
            "output_token_ids": [], 
            "test_result": []
        }
        self.result_dict = {}
    
    def print_metric(self):
        data = [
            [
                self.task_config.task_name,
                self.task_config.shots,
                self.case_num_list[0],
                "truthfulqa",
                self.result_dict.get(MC1),
                self.result_dict.get(MC2),
                self.result_dict.get(BLUE_DIFF),
                self.result_dict.get(ROUGE1_DIFF)
            ]
        ]
        for idx, sub_dataset in enumerate(self.task_config.subject_mapping.keys()):
            data.append([self.task_config.subject_mapping[sub_dataset]["name"],
                         self.task_config.shots,
                         self.case_num_list[idx],
                         "truthfulqa",
                         self.result_dict.get(MC1),
                         self.result_dict.get(MC2),
                         self.result_dict.get(BLUE_DIFF),
                         self.result_dict.get(ROUGE1_DIFF)])
            break
        headers = ["Tasks", "N-Shot", "Total", "Metric", "MC1", "MC2", "BLUE", "ROUGE"]
        markdown_table = tabulate(data, headers, tablefmt="grid")
        logger.info(f"Metric Table:\n{markdown_table}")
        with safe_open(self.result_file_path, mode='w', encoding='utf-8') as file:
            file.write(f"runner config:\n{self.runner_config}\n")
            file.write(f"model config:\n{self.model_config}\n")
            file.write(f"task config:\n{self.task_config}\n")
            file.write(f"metric table:\n{markdown_table}\n")
        super().print_metric()