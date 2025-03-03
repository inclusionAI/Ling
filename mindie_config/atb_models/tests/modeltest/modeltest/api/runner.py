# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass, asdict
from functools import wraps
import math
from tabulate import tabulate
import torch
from tqdm import tqdm
from modeltest.task.precision_task import PrecisionTask
from modeltest.task import get_task_cls
from modeltest.metric import get_metric_cls
from atb_llm.utils.log.logging import logger
from .task import Task
from .model import Model
from .metric import Metric


@dataclass
class RunnerConfig:
    model_config_path: str
    task_config_path: str
    tp: str
    batch_size: bool
    save_debug_enable: bool

    def __repr__(self):
        config_table = [[k, v] for k, v in asdict(self).items()]
        return tabulate(config_table, headers=["Field", "Value"], tablefmt="grid")


class Runner():
    def __init__(self, device_type, runner_config, output_dir) -> None:
        self.device_type = device_type
        self.runner_config = runner_config
        self.task: Task = get_task_cls(self.runner_config.task_config_path)
        self.model_config = Model.parse_config(self.runner_config.model_config_path)
        self.output_dir = output_dir
        self.model_runner = None
        self.tokenizer = None
        self.metric = None

    @staticmethod
    def on_device_0_only(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.get_rank() == 0:
                return func(self, *args, **kwargs)
        return wrapper

    def get_rank(self):
        raise NotImplementedError("Subclasses should implement get_rank.")

    def post_init(self):
        self.metric: Metric = get_metric_cls(self.task.task_config.metric_type)(
            self.task.task_config,
            self.model_runner.model_config,
            self.device_type,
            self.runner_config,
            self.output_dir)
        self.task.set_tokenizer(self.tokenizer)

    def start(self):
        getattr(self, f"{self.task.task_config.task_type}_task_evaluate")()

    def get_task_choice(self):
        pass

    def run_inference(self, queries):
        raise NotImplementedError("Subclasses should implement run_inference.")

    @on_device_0_only
    def save_queries_and_token_ids(self, *args):
        self.save_queries_and_token_ids_impl(*args)

    def save_queries_and_token_ids_impl(self, queries, result_tuple):
        raise NotImplementedError("Subclasses should implement save_queries_and_token_ids_impl.")

    @on_device_0_only
    def get_logits(self, result_tuple):
        if self.task.task_config.need_logits:
            return self.get_logits_impl(result_tuple)
        return torch.tensor([])

    def get_logits_impl(self, result_tuple):
        raise NotImplementedError("Subclasses should implement get_logits_impl.")

    @on_device_0_only
    def result_judge(self, *args):
        self.task.result_judge(self.metric, *args)

    @on_device_0_only
    def save_results(self):
        self.metric.print_metric()

    @on_device_0_only
    def save_debug_info(self):
        if self.runner_config.save_debug_enable:
            self.metric.save_debug()

    def precision_task_evaluate(self):
        if self.task is not None and not isinstance(self.task, PrecisionTask):
            raise TypeError("Expected an instance of PrecisionTask.")
        datasets_input = self.task.prepare_data(self.metric)

        with torch.no_grad():
            for sub_dataset_idx, sub_dataset in enumerate(datasets_input):
                logger.info("subdataset %d / %d, task name: %s START: ", sub_dataset_idx + 1, len(datasets_input),
                            list(self.task.task_config.subject_mapping.keys())[sub_dataset_idx])
                self.metric.case_num += len(sub_dataset)
                self.metric.case_num_list.append(len(sub_dataset))
                for batch_idx, batched_data in tqdm(self.task.get_batched_data(sub_dataset,
                                                    batch_size=self.runner_config.batch_size),
                                                    total=math.ceil(len(sub_dataset) // self.runner_config.batch_size)):
                    try:
                        queries = self.task.build_queries(sub_dataset_idx, batched_data, self.model_config)
                        result_tuple = self.run_inference(queries)
                        self.save_queries_and_token_ids(queries, result_tuple)
                        logits = self.get_logits(result_tuple) # optional
                        self.result_judge(result_tuple.generate_text, logits, sub_dataset_idx, batched_data)
                    except Exception as e:
                        self.metric.error_num += 1
                        self.metric.error_list.append(tuple([sub_dataset_idx, batch_idx]))
                        logger.error("Error occurred %s", str(e), exc_info=True)
                logger.info("subdataset %d / %d, task name: %s END. ", sub_dataset_idx + 1, len(datasets_input),
                            list(self.task.task_config.subject_mapping.keys())[sub_dataset_idx])
        self.save_results()
        self.save_debug_info()


def get_runner_cls():
    if torch.cuda.is_available():
        from modeltest.runner.gpu_runner import GPURunner
        return GPURunner
    else:
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                from modeltest.runner.npu_runner import NPURunner
                return NPURunner
        except ImportError as e:
            pass
    raise RuntimeError("Modeltest only support running on GPU/NPU") from e
