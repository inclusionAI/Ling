# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from functools import wraps
import os
import datetime
import torch
import pytz
from modeltest.model.npu_model import NPUModel
from modeltest.api.runner import Runner
from atb_llm.utils.file_utils import safe_open


class NPURunner(Runner):
    def __init__(self, *args) -> None:
        super().__init__('NPU', *args)
        self.set_environ()
        self.model_runner = NPUModel(self.runner_config.batch_size, self.model_config, self.task.task_config)
        self.model = self.model_runner.model
        self.tokenizer = self.model.tokenizer
        self.now_str = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H:%M:%S.%f")
        self.post_init()

    @staticmethod
    def enable_logits_save(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
            os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.metric.data_dir
            rtv = func(self, *args, **kwargs)
            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
            return rtv
        return wrapper

    @staticmethod
    def enable_token_ids_save(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            os.environ['ATB_LLM_TOKEN_IDS_SAVE_ENABLE'] = "1"
            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.metric.data_dir
            rtv = func(self, *args, **kwargs)
            os.environ['ATB_LLM_TOKEN_IDS_SAVE_ENABLE'] = "0"
            return rtv
        return wrapper
    
    def set_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = "expandable_segments:True"
        for env_name, env_value in self.model_config.env.items():
            os.environ[env_name] = env_value
    
    def get_rank(self):
        return self.model.rank
    
    def save_queries_and_token_ids_impl(self, queries, result_tuple):
        if self.model_config.mm_model:
            with safe_open(os.path.join(self.metric.debug_dir, f"outputs_{self.now_str}.txt"), 'a') as f:
                f.write(str(result_tuple.generate_text))
                f.write(str(queries))
                f.write(str(result_tuple.e2e_time) + '\n')
        else:
            for i, query in enumerate(queries):
                self.metric.csv_debug.get("key", []).append(len(self.metric.csv_debug.get("key", [])))
                self.metric.csv_debug.get("queries", []).append(query)
                input_token_ids = torch.load(os.path.join(self.metric.data_dir, f'input_ids_{i}.pth'))
                self.metric.csv_debug.get("input_token_ids", []).append(input_token_ids.tolist())
                with safe_open(os.path.join(self.metric.data_dir, f"output_ids_{i}.txt"), 'r') as f:
                    output_token_ids = list(map(int, f.read().split()))
                self.metric.csv_debug.get("output_token_ids", []).append(output_token_ids)   
    
    def get_logits_impl(self, _):
        return torch.load(os.path.join(self.metric.data_dir, 'logits_0.pth'))

    @enable_token_ids_save
    @enable_logits_save
    def run_inference(self, queries):
        if self.model_runner is not None and not isinstance(self.model_runner, NPUModel):
            raise TypeError("Expected a model of NPUModel.")
        extra_args = self.task.get_npu_runner_extra_args()
        return self.model_runner.inference(
            queries,
            self.runner_config.batch_size,
            self.task.task_config.requested_max_output_length,
            False,
            **extra_args)
