# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from dataclasses import dataclass, asdict
from typing import Union, List, Dict
import torch
import yaml
from tabulate import tabulate
from atb_llm.utils.log.logging import logger
from atb_llm.utils.file_utils import safe_open


@dataclass
class TaskConfig:
    task_type: str
    task_name: str
    hf_dataset_path: str
    om_dataset_path: str
    local_dataset_path: str
    prompt: str
    choices: List
    shots: int
    requested_max_input_length: int
    requested_max_output_length: int
    need_logits: bool
    need_truncate_input: bool
    metric: Dict[str, Union[str, float]]
    metric_type: str
    metadata_version: str
    humaneval_x_datasets_selector: List[str]
    subject_mapping: Dict

    def __repr__(self):
        config_table = [[k, v] for k, v in asdict(self).items()]
        return tabulate(config_table, headers=["Field", "Value"], tablefmt="grid", maxcolwidths=[None, 100])


class Task():
    def __init__(self, task_config) -> None:
        self.task_config: TaskConfig = task_config
        self.tokenizer = None
        self.local_dataset_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.metric_type = task_config.metric.get('metric_type', 'pass_k')
        self.k_value = task_config.metric.get('k', 1.0)

    @staticmethod
    def parse_config(config_path):
        with safe_open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
            task_config = TaskConfig(
                task_type=config_dict.get('task_type', ""),
                task_name=config_dict.get('task_name', ""),
                hf_dataset_path=config_dict.get('hf_dataset_path', ""),
                om_dataset_path=config_dict.get('om_dataset_path', ""),
                local_dataset_path=config_dict.get('local_dataset_path', ""),
                prompt=config_dict.get('prompt', ""),
                choices=config_dict.get('choices', []),
                shots=config_dict.get('shots', 0),
                requested_max_input_length=config_dict.get('requested_max_input_length', 256),
                requested_max_output_length=config_dict.get('requested_max_output_length', 256),
                need_logits=config_dict.get('need_logits', False),
                need_truncate_input=config_dict.get('need_truncate_input', False),
                metric=config_dict.get('metric', {}),
                metric_type=config_dict.get('metric', {}).get('metric_type', ""),
                metadata_version=config_dict.get('metadata', {}).get('version', "1.0"),
                humaneval_x_datasets_selector=config_dict.get('humaneval_x_datasets_selector', []),
                subject_mapping=config_dict.get('subject_mapping', {})
            )
        
        if not os.path.exists(task_config.local_dataset_path):
            task_config.local_dataset_path = os.path.join("..", task_config.local_dataset_path)
        logger.info(f"task config:\n{task_config}")
        return task_config
    
    @staticmethod
    def get_npu_runner_extra_args():
        return {}

    def run(self):
        with torch.no_grad():
            self.inference()

    def inference(self):
        pass

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_data(self, metric):
        raise NotImplementedError("Subclasses should implement prepare_data.")

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        raise NotImplementedError("Subclasses should implement build_queries.")

    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        raise NotImplementedError("Subclasses should implement result_judge.")
