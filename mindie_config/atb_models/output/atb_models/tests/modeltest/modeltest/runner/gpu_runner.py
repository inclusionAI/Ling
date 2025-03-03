# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import importlib
import torch
from modeltest.model.gpu_model import GPUModel
from modeltest.api.runner import Runner


class GPURunner(Runner):
    def __init__(self, *args) -> None:
        super().__init__('GPU', *args)
        self.backend = self.model_config.requested_gpu_framework
        self.model_runner = self.get_model_runner_cls()(
            self.runner_config.tp,
            self.model_config,
            self.task.task_config)
        self.model = self.model_runner.model
        self.tokenizer = self.model_runner.tokenizer
        self.post_init()
    
    def get_rank(self):
        return torch.cuda.current_device()
    
    def get_model_runner_cls(self):
        module = importlib.import_module(f"modeltest.model.{self.backend.lower()}_model")
        return getattr(module, f"{self.backend}Model")
    
    def run_inference(self, queries):
        if self.model_runner is not None and not isinstance(self.model_runner, GPUModel):
            raise TypeError("Expected a model of GPUModel.")
        return self.model_runner.inference(
            queries, self.task.task_config.requested_max_output_length)
    
    def save_queries_and_token_ids_impl(self, queries, result_tuple):
        for _, query in enumerate(queries):
            self.metric.csv_debug.get("key", []).append(len(self.metric.csv_debug.get("key", [])))
            self.metric.csv_debug.get("queries", []).append(query)
        self.metric.csv_debug.get("input_token_ids", []).extend(result_tuple.input_id)
        self.metric.csv_debug.get("output_token_ids", []).extend(result_tuple.generate_id)
    
    def get_logits_impl(self, result_tuple):
        return result_tuple.logits