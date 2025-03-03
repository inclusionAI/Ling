# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass, asdict
from tabulate import tabulate
import yaml
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log.logging import logger
from .task import TaskConfig


@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    model_type: str
    data_type: str
    use_chat_template: bool
    max_position_embedding: int
    block_size: int
    requested_gpu_framework: str
    trust_remote_code: bool
    env: dict
    mm_model: dict

    def __post_init__(self):
        if self.requested_gpu_framework not in ["Huggingface", "Vllm"]:
            raise RuntimeError("requested_gpu_framework in model config yaml must choose from Huggingface and Vllm.")

    def __repr__(self):
        config_table = [[k, v] for k, v in asdict(self).items()]
        return tabulate(config_table, headers=["Field", "Value"], tablefmt="grid")


@dataclass
class ResultMetadata:
    generate_text: list
    generate_id: list
    logits: list
    input_id: list
    token_num: list
    e2e_time: float


class Model():
    def __init__(self, device_type, model_config, task_config: TaskConfig) -> None:
        self.device_type = device_type
        self.model_config: ModelConfig = model_config
        self.task_config = task_config
        self.model = None

    @staticmethod
    def parse_config(config_path):
        with safe_open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
            model_config = ModelConfig(
                model_name=config_dict.get('model_name', ""),
                model_path=config_dict.get('model_path', ""),
                model_type=config_dict.get('model_type', ""),
                data_type=config_dict.get('data_type', ""),
                use_chat_template=config_dict.get('use_chat_template', False),
                max_position_embedding=config_dict.get('max_position_embedding', -1),
                block_size=config_dict.get('block_size', 128),
                requested_gpu_framework=config_dict.get('requested_gpu_framework', "Huggingface"),
                trust_remote_code=config_dict.get('trust_remote_code', False),
                env=config_dict.get('env', {}),
                mm_model=config_dict.get('mm_model', None),
            )
        logger.info(f"model config:\n{model_config}")
        return model_config
