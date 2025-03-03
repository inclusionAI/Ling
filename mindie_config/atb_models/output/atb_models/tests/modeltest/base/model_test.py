# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import csv
import sys
import glob
import json
import logging
import math
import os
import re
import stat
import shutil
import time
import argparse
import ast
import traceback
from datetime import datetime, timedelta, timezone
from importlib import reload
from pathlib import Path
from decimal import Decimal, InvalidOperation

import torch
import torch.nn.functional as F
try:
    import torch_npu
except ModuleNotFoundError:
    pass
try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    pass
except ImportError:
    pass
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationConfig
from transformers.configuration_utils import PretrainedConfig
from tqdm import tqdm
from atb_llm.utils.file_utils import standardize_path, check_file_safety, safe_open
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper
try:
    ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
    sys.path.append(os.path.join(ATB_SPEED_HOME_PATH, "../.."))
    sys.path.append(ATB_SPEED_HOME_PATH)
    from atb_llm.utils import env
    from atb_llm.utils.file_utils import safe_open
    from examples.run_pa import PARunner
    from examples.run_fa import FARunner
except TypeError:
    pass
from dataset.full.CMMLU.categories import name_en2zh
from dataset.full.CMMLU import cmmlu_eval
from dataset.full.HumanEval_X import humaneval_x_utils
from dataset.full.HumanEval_X import humaneval_x_eval
from dataset.full.NeedleBench import needlebench_single
from dataset.full.TruthfulQA import truthfulqa_eval
from dataset.full.HumanEval.human_eval import evaluate_functional_correctness
from dataset.full.LongBench.longbench_eval import scorer, scorer_e
from atb_llm.utils.log.logging import logger, print_log

QA_PRIMER = """Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain."""

UTILS_CODE_MARKER = "    def greedy_search(\n"

UTILS_CODE_INSERTED_PART_1 = """
        import os
        import time
        if os.environ.get('test_mode') != '':
            tensor_folder = os.environ.get('tensor_folder')
            if tensor_folder is not None:
                os.makedirs(tensor_folder, exist_ok=True)
                if not os.path.exists(tensor_folder):
                    raise RuntimeError(f"folder {tensor_folder} create fail")
            else:
                raise RuntimeError(f"tensor_folder env not exist")
        cnt = 0
        first_token_time = 0
        non_first_token_time = 0
"""
UTILS_CODE_INSERTED_PART_2 = """
            getattr(torch, os.environ.get('core_type')).synchronize()
            forward_start_time = time.time()
"""
UTILS_CODE_INSERTED_PART_3 = """
            if os.environ.get('test_mode') == 'simplified':
                tensor_folder = os.environ.get('tensor_folder')
                if torch.distributed.get_rank() == 0:
                    torch.save(next_token_logits.cpu(), f"{tensor_folder}/logits_{cnt}.pth")
                    torch.save(next_tokens.cpu(), f"{tensor_folder}/tokens_{cnt}.pth")
"""
UTILS_CODE_INSERTED_PART_4 = """
            getattr(torch, os.environ.get('core_type')).synchronize()
            forward_end_time = time.time()
            if cnt != 0:
                non_first_token_time += (forward_end_time - forward_start_time)
            else:
                first_token_time = forward_end_time - forward_start_time
            cnt += 1    
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            first_token_time_tensor = torch.tensor([first_token_time])
            non_first_token_time_tensor = torch.tensor([non_first_token_time])
            torch.save(first_token_time_tensor.cpu(), f"{tensor_folder}/first_token_time.pth")
            torch.save(non_first_token_time_tensor.cpu(), f"{tensor_folder}/non_first_token_time.pth")
"""

UTILS_CODE_INSERTED_MARKER = "        import os\n"

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")

communication_map = {"NPU": "hccl", "GPU": "nccl"}
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
core_map = {"NPU": "npu", "GPU": "cuda"}
prompt_map = {"GSM8K": "", "TruthfulQA": QA_PRIMER}
question_num = {"GSM8K": 11, "TruthfulQA": 12}

logging.basicConfig(level=logging.DEBUG)

KEY = "key"
QUERIES = "queries"
LENGTH = "length"
PASS = "pass"
ALL_CLASSES = "all_classes"
OUTPUT_TOKEN_IDS = "output_token_ids"
INPUT_TOKEN_IDS = "input_token_ids"
TEST_RESULT = "test_result"
GOLDEN_RESULT = "golden_result"
CLEANED_UP_RESULTS = "cleaned_up_results"
ATB_LLM_TOKEN_IDS_SAVE_ENABLE = "ATB_LLM_TOKEN_IDS_SAVE_ENABLE"
TASK_ID = "task_id"
PROMPT = "prompt"
PT = "pt"
LONGBENCH = "LongBench"
NEEDLEBENCH = "NeedleBench"
QWEN = "qwen"


class ModelTest:
    def __init__(self, model_type, data_type, test_mode, model_name, output_dir, dataset_name, batch_size_lst, 
                 device_id, hardware_type, case_pair, weight_dir, time_limit, max_position_embedding,
                 input_text_or_file, is_chat_model, is_instruct_model, shot, lora_adapter, context_length, kw_args,
                 prefill_batch_size, trust_remote_code) -> None:
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.model_type = model_type
        self.data_type = data_type
        self.test_mode = test_mode
        self.model_name = model_name
        self.dataset_name = dataset_name               
        self.shot = shot
        self.lora_adapter = lora_adapter
        self.batch_size_lst = self.__parse_bs(batch_size_lst)
        self.prefill_batch_size = prefill_batch_size
        self.device_id = device_id
        self.hardware_type = hardware_type
        self.block_size = self.get_block_size()
        self.case_pair = ast.literal_eval(case_pair) if case_pair != "[]" else [[256, 256], [512, 512], 
                                                                                [1024, 1024], [2048, 2048]]
        self.local_case_pair = self.case_pair
        self.input_text_or_file = input_text_or_file
        self.weight_dir = weight_dir
        self.time_limit = time_limit
        self.max_position_embedding = max_position_embedding
        self.is_chat_model = is_chat_model
        self.is_instruct_model = is_instruct_model
        self.core_type = core_map.get(self.hardware_type, "npu")
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_format_nz = False
        self.quantize = None
        self.result_dir = ""
        self.data_dir = ""
        self.log_dir = ""
        self.debug_dir = ""
        self.output_dir = output_dir
        self.current_result_path = ''
        self.causal_bs = 0
        self.error_catched = 0
        self.error_rate = 0
        self.error_message = ""
        self.model = None
        self.tokenizer_wrapper = TokenizerWrapper(model_name_or_path=self.weight_dir) if is_chat_model else None
        self.csv_debug = {}
        self.logger = logger
        self.context_length = context_length
        self.csv_path = None
        self.csv_formatted_path = None
        self.fa_runner = None
        self.pa_runner = None
        self.fa_runner = None
        self.tokenizer = None
        self.tokenizer_params = None
        self.eos_token = None
        self.dataset_path = None
        self.dataset_list = None
        self.test_type = None
        self.formatted_datetime = None
        self.batch_size = 0
        self.satisfy_time_limit = 0
        self.kw_args = kw_args
        self.warmup_seq_in = 0
        self.warmup_seq_out = 0
        self.warmup_bs = 0
        self.trust_remote_code = trust_remote_code

    @staticmethod
    def get_chip_num():
        return 1
    
    @staticmethod
    def get_dataset_list():
        return ["GSM8K", "TruthfulQA", "MMLU", "CEval", "BoolQ", "NeedleBench"]
    
    @staticmethod
    def clear():
        os.unsetenv("test_mode")
        os.unsetenv("hardware_type")
        os.unsetenv("tensor_folder")

    @staticmethod
    def _postprocess(text: str, options: str, cushion=True) -> str:
        patterns = [
            f'答案是?\s?([{options}])',
            f'答案是?\s?：([{options}])',
            f'答案是?\s?:([{options}])',
            f'答案应该?是\s?([{options}])',
            f'答案应该?选\s?([{options}])',
            f'答案为\s?([{options}])',
            f'答案选\s?([{options}])',
            f'选择?\s?([{options}])',
            f'故选?\s?([{options}])'
            f'只有选?项?\s?([{options}])\s?是?对',
            f'只有选?项?\s?([{options}])\s?是?错',
            f'只有选?项?\s?([{options}])\s?不?正确',
            f'只有选?项?\s?([{options}])\s?错误',
            f'说法不?对选?项?的?是\s?([{options}])',
            f'说法不?正确选?项?的?是\s?([{options}])',
            f'说法错误选?项?的?是\s?([{options}])',
            f'([{options}])\s?是正确的',
            f'([{options}])\s?是正确答案',
            f'选项\s?([{options}])\s?正确',
            f'所以答\s?([{options}])',
            f'所以\s?([{options}][.。$]?$)',
            f'所有\s?([{options}][.。$]?$)',
            f'[\s，：:,]([{options}])[。，,\.]?$',
            f'[\s，,：:][故即]([{options}])[。\.]?$',
            f'[\s，,：:]因此([{options}])[。\.]?$',
            f'[是为。]\s?([{options}])[。\.]?$',
            f'因此\s?([{options}])[。\.]?$',
            f'显然\s?([{options}])[。\.]?$',
            '答案是\s?(\S+)(?:。|$)',
            '答案应该是\s?(\S+)(?:。|$)',
            '答案为\s?(\S+)(?:。|$)',
            f'[Tt]he answer is \(?([{options}])\)?',
            f'[Tt]he answer is option \(?([{options}])\)?',
            f'[Tt]he correct answer is \(?([{options}])\)?',
            f'[Tt]he correct answer is option \(?([{options}])\)?',
            f'[Tt]he answer to the question is \(?([{options}])\)?',
            f'^选项\s?([{options}])',
            f'^([{options}])\s?选?项',
            f'(\s|^)[{options}][\s。，,：:\.$]',
            f'(\s|^)[{options}](\s|$)',
            '1.\s?(.*?)$',
            f'1.\s?([{options}])[.。$]?$',
        ]
        cushion_patterns = [
            f'([{options}]):',
            f'[{options}]',
        ]

        if cushion:
            patterns.extend(cushion_patterns)
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                outputs = match.group(0)
                for i in options:
                    if i in outputs:
                        return i
        return ''

    @staticmethod
    def __parse_bs(batch_size_lst):
        try:
            batch_size_lst = int(batch_size_lst)
            return [batch_size_lst]
        except ValueError:
            pass

        try:
            batch_size_list = [int(bs) for bs in batch_size_lst.split(',')]
            batch_size_lst = batch_size_list
            return batch_size_lst
        except ValueError:
            pass
        
        try:
            batch_size_lst = ast.literal_eval(batch_size_lst)
            if isinstance(batch_size_lst, list):
                if not batch_size_lst:
                    raise ValueError("Batchsize input is empty")
                else:
                    return batch_size_lst
            raise ValueError("Wrong batchsize input format")
        except (ValueError, SyntaxError) as e:
            raise ValueError("Wrong batchsize input format") from e
    
    @classmethod
    def create_instance(cls):
        args = get_args()
        test_instance = cls(*args)
        test_instance.run()
    
    def get_block_size(self):
        return 128

    def run(self):
        if self.test_mode == "performance_maxbs":
            self.__run_maxbs()
        else:
            self.__run_multibs()

    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'revision': None,
            'use_fast': True,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': self.trust_remote_code
        }

    def prepare_environ(self):
        pass
    
    def get_supported_model_type(self):
        pass

    def calculate_max_memory_requirements(self):
        max_memory = -1
        max_memory_case_pair = None
        max_bs = None

        for batch_sizes in self.batch_size_lst:
            for batch_size in batch_sizes:
                for seq_in, seq_out in self.case_pair:
                    memory = batch_size * (seq_in + seq_out)
                    if memory > max_memory:
                        max_memory = memory
                        max_bs = batch_size
                        max_memory_case_pair = (seq_in, seq_out)

        return max_bs, max_memory_case_pair

    def __run_maxbs(self):
        self.batch_size = self.batch_size_lst[0]
        self.local_case_pair = [self.case_pair]
        self.satisfy_time_limit = 1
        self.__run_single_bs()
        messages = [
            self.error_catched,
            self.satisfy_time_limit,
            self.csv_path
        ]
        if self.rank == 0:
            with safe_open(os.path.join(self.script_path, "../maxbs.txt"), "w") as file:
                for message in messages:
                    file.write(str(message) + "\n")
            
    def __run_multibs(self):
        if not isinstance(self.batch_size_lst[0], list):
            for i in self.batch_size_lst:
                self.batch_size = i
                self.__run_single_bs()
        else:
            if self.test_mode == "performance_single" or self.test_mode == "precision_single":
                raise ValueError("performance_single or precision_single mode only support one batchsize")
            if self.test_mode != "performance":
                raise ValueError("only performance test support causal bs input")
            max_bs, max_memory_case_pair = self.calculate_max_memory_requirements()
            seq_in, seq_out = max_memory_case_pair
            bs_lst_len = len(self.batch_size_lst)
            if len(self.case_pair) != bs_lst_len:
                raise ValueError("inconsistent case_pair and batch_size input, length should be the same")
            
            self.causal_bs = 1
            self.local_case_pair = self.case_pair
            arranged_lst = [[0] * bs_lst_len for _ in range(max_bs)]

            for idx, sublst in enumerate(self.batch_size_lst):
                for bs in sublst:
                    arranged_lst[bs - 1][idx] = 1
            self.batch_size_lst = arranged_lst

            self.warmup_seq_in = seq_in
            self.warmup_seq_out = seq_out
            self.warmup_bs = max_bs

            for bs, sublst in enumerate(self.batch_size_lst):
                if sum(sublst) != 0:
                    self.batch_size = bs + 1
                    if not self.error_catched:
                        self.__run_single_bs()

    def __run_single_bs(self):
        self.prepare_environ()
        self.__prepare_and_check()
        self.__run()
        self.clear()

    def __prepare_and_check(self):
        performance = "performance"
        simplified = "simplified"
        cst_timezone = timezone(timedelta(hours=8))
        current_time = datetime.now(cst_timezone)
        self.formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    
        if performance in self.test_mode:
            self.test_type = performance
        elif simplified in self.test_mode:
            self.test_type = simplified
        else:
            self.test_type = "precision"
    
        max_csv_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_csv_limit)
                break
            except OverflowError:
                max_csv_limit = int(max_csv_limit / 10)
        
        config_path = os.path.join(self.weight_dir, "config.json")
        with safe_open(config_path, 'r') as f:
            config_data = json.load(f)
            if "quantize" in config_data:
                self.quantize = config_data["quantize"]

        if self.quantize:
            csv_base_path = os.path.join(f"{self.data_type}_{self.quantize}", self.model_name)
        else:
            csv_base_path = os.path.join(f"{self.data_type}", self.model_name)
        
        self.result_dir = os.path.join(self.output_dir, "results", self.hardware_type, 
                                       f"{self.test_type}_test", self.test_mode, csv_base_path)
        self.data_dir = os.path.join(self.output_dir, "data", self.hardware_type, 
                                     f"{self.test_type}_test", self.test_mode, csv_base_path)
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.debug_dir = os.path.join(self.output_dir, "debug", self.hardware_type, 
                                      f"{self.test_type}_test", self.test_mode, csv_base_path)

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.__create_folder(self.data_dir)

        if self.test_type == performance:
            if "performance_maxbs" in self.test_mode:
                self.csv_path = os.path.join(self.result_dir, 
                                             f"{self.test_mode}_{self.model_type}_{self.case_pair[0]}_"
                                             f"{self.case_pair[1]}_batch{self.batch_size}_"
                                             f"tp{self.world_size}_result.csv")
                self.csv_formatted_path = os.path.join(self.result_dir, 
                                                       f"{self.test_mode}_{self.model_type}_{self.case_pair[0]}_"
                                                       f"{self.case_pair[1]}_batch{self.batch_size}_"
                                                       f"tp{self.world_size}_formatted_result.csv")
            else:
                self.csv_path = os.path.join(self.result_dir, 
                                             f"{self.test_mode}_{self.model_type}_batch{self.batch_size}_"
                                             f"tp{self.world_size}_result.csv")
                self.csv_formatted_path = os.path.join(self.result_dir, 
                                                       f"{self.test_mode}_{self.model_type}_batch{self.batch_size}_"
                                                       f"tp{self.world_size}_formatted_result.csv")
        else:
            self.csv_path = os.path.join(self.result_dir, 
                                         f"{self.dataset_name}_{self.model_type}_batch{self.batch_size}_"
                                         f"tp{self.world_size}_{self.test_mode}_result.csv")
            self.csv_formatted_path = os.path.join(self.result_dir, 
                                                   f"{self.dataset_name}_{self.model_type}_batch{self.batch_size}_"
                                                   f"tp{self.world_size}_{self.test_mode}_formatted_result.csv")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(self.csv_formatted_path, flags, modes), 'w', encoding='utf-8') as f:
            if self.test_mode.startswith(performance):
                f.write("{:<15s}|{:<15s}|{:<15s}|{:<15s}|{:<15s}|{:<25s}|{:<25s}|"
                        "{:<36s}|{:<25s}|{:<45s}|{:<35s}\n".format(
                            "Model", "Batchsize", "In_seq", "Out_seq", "Total time(s)", "First token time(ms)", 
                            "Non-first token time(ms)", "Non-first token Throughput(Tokens/s)", 
                            "E2E Throughput(Tokens/s)", "Non-first token Throughput Average(Tokens/s)", 
                            "E2E Throughput Average(Tokens/s)"
                        )
                )
            elif self.test_mode == simplified:
                f.write("Standard: [1] KL loss <= 1e-3. [2] rate of KL loss > 1e-4 <= 0.5%.\n")
                f.write("{:<15s}|{:<15s}|{:<15s}|{:<15s}|{:<15s}|{:<15s}|{:<15s}\n".format(
                    "Model", "Dataset", "Batchsize", "Logits Num", "Greatest KLL", "Error Rate", "Result"
                ))

        self.logger.info(
            "\nmodel_name: %s\nmodel_type: %s\ndata_type: %s\ntest_mode: %s\n"
            "data_dir: %s\ntime_limit: %d\nbatch_size: %d\nresult_dir: %s\nlog_dir: %s", 
            self.model_name, self.model_type, self.data_type, 
            self.test_mode, self.data_dir, self.time_limit, 
            self.batch_size, self.result_dir, self.log_dir)
        
        config_dict, _ = PretrainedConfig.get_config_dict(self.weight_dir)
        model_spec = config_dict.get('model_type')
        if model_spec not in self.get_supported_model_type():
            raise RuntimeError("input model name not match model type in config file, "
                               "please check file in core folder")

        if self.hardware_type == "NPU":
            reload(env)
            if ATB_HOME_PATH is None:
                self.logger.error("env ATB_HOME_PATH not exist, source atb set_env.sh")
                raise RuntimeError(
                    "env ATB_HOME_PATH not exist, source atb set_env.sh")
            self.logger.info("ATB env get success.")
            if ATB_SPEED_HOME_PATH is None:
                self.logger.error("env ATB_SPEED_HOME_PATH not exist, source atb_speed set_env.sh")
                raise RuntimeError(
                    "env ATB_SPEED_HOME_PATH not exist, source atb_speed set_env.sh")
            self.logger.info("ATB_SPEED env get success")

        if self.test_mode == "performance_single" or self.test_mode == "precision_single":
            try:
                input_text_or_file = ast.literal_eval(self.input_text_or_file)
                self.input_text_or_file = input_text_or_file
            except ValueError:
                with safe_open(self.input_text_or_file, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                self.input_text_or_file = [line for line in lines]
            if len(self.input_text_or_file) > self.batch_size:
                self.logger.warning("input texts length exceeds the input batch_size, "
                                    "please check if it's not on your purpose.")
        if self.test_mode == "full":
            self.dataset_list = self.get_dataset_list()
            if self.dataset_name not in self.dataset_list:
                self.logger.warning("%s not support %s, please check", self.model_name, self.dataset_name)

        os.environ['core_type'] = self.core_type
        os.environ['test_mode'] = self.test_mode
        torch.manual_seed(1)

    def __run(self):
        test_mode_list = ["simplified", "full", "precision_single", "lite"]
        if self.test_mode in test_mode_list:
            self.__run_precision()
        elif self.test_mode.startswith("performance"):
            self.__run_performance()
        else:
            self.logger.error("%s test not support, only support performance*, simplified and full", self.test_mode)
            raise RuntimeError(f"{self.test_mode} test not support, only support performance*, simplified and full")

    def __run_performance(self):
        self.logger.info("performance test start")
        performance_prompt = [
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
        ]

        csv_results = []
        if self.hardware_type == "GPU":
            os.environ['tensor_folder'] = self.data_dir

        def warmup():
            self.logger.info("performance test warmup start")
            if self.model_type == "fa" and self.hardware_type == "GPU":
                warmup_input_ids = torch.randint(0, self.model.config.vocab_size, [self.batch_size, 2048],
                                                 dtype=torch.int64)
                warmup_attention_mask = torch.ones((self.batch_size, 2048), dtype=torch.int64)
                inputs = self.tokenizer(performance_prompt * self.batch_size, return_tensors=PT, padding='max_length',
                                        max_length=2048)
                inputs["input_ids"] = warmup_input_ids
                inputs["attention_mask"] = warmup_attention_mask

                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                with torch.no_grad():
                    _ = self.model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=4,
                        eos_token_id=self.model.config.vocab_size * 2
                    )
            else:
                pass
            self.logger.info("performance test warmup end")

        def run_performance_test():
            non_first_token_throughput_total = 0
            e2e_throughput_total = 0
            passed_cases = 0
            if self.causal_bs:
                filtered_case_pair = list(
                    filter(lambda x: x[1] == 1, 
                           zip(
                               self.case_pair, 
                               self.batch_size_lst[self.batch_size - 1]
                            )
                    )
                )
                self.local_case_pair = [i[0] for i in filtered_case_pair]
            for seq_len_in, seq_len_out in self.local_case_pair:
                self.error_catched = 0
                self.error_message = ""
                self.logger.info("batch_size: %d, seq_len_in: %d, seq_len_out: %d", 
                                 self.batch_size, seq_len_in, seq_len_out)
                if self.model_type == "fa":
                    if self.hardware_type == "GPU":
                        input_ids = torch.randint(0, self.model.config.vocab_size, [self.batch_size, seq_len_in],
                                                dtype=torch.int64)
                        attention_mask = torch.ones((self.batch_size, seq_len_in), dtype=torch.int64)
                        inputs = self.tokenizer(performance_prompt * self.batch_size, return_tensors=PT,
                                                padding='max_length',
                                                max_length=seq_len_in)
                        inputs["input_ids"] = input_ids
                        inputs["attention_mask"] = attention_mask

                        input_ids = inputs.input_ids.to(self.model.device)
                        attention_mask = inputs.attention_mask.to(self.model.device)

                        with torch.no_grad():
                            getattr(torch, self.core_type).synchronize()
                            e2e_start = time.time()
                            generate_ids = self.model.generate(inputs=input_ids,
                                                            attention_mask=attention_mask,
                                                            min_new_tokens=seq_len_out,
                                                            max_new_tokens=seq_len_out
                                                            )
                            try:
                                _ = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)
                            except Exception as e:
                                self.logger.warning("Decoding failed, falling back to individual decoding. Error: %s", 
                                                    str(e))
                                _ = [
                                    self.tokenizer.decode(output)
                                    for output in generate_ids[:, inputs["input_ids"].size(1):].tolist()
                                ]
                            getattr(torch, self.core_type).synchronize()
                            e2e_end = time.time()
                            e2e_time = e2e_end - e2e_start
                    else:
                        self.__get_model_or_runner(seq_len_in, seq_len_out)
                        try:
                            self.fa_runner.warm_up()
                    
                            input_ids = torch.randint(0, 32000, [self.batch_size, seq_len_in],
                                                dtype=torch.int64)
                            attention_mask = torch.ones((self.batch_size, seq_len_in), dtype=torch.int64)
                        
                            _, _, e2e_time = self.fa_runner.infer(input_ids, attention_mask)
                        
                            passed_cases += 1
                        except Exception as e:
                            self.error_catched = 1
                            self.error_message = str(e)
                            self.logger.error("error catched: %s", self.error_message)
                        del self.fa_runner
                        torch.npu.empty_cache()                        
                else:
                    try:
                        if self.pa_runner is None:
                            seq_len_in_pa = self.warmup_seq_in if self.warmup_seq_in != 0 else seq_len_in
                            seq_len_out_pa = self.warmup_seq_out if self.warmup_seq_out != 0 else seq_len_out
                            self.__get_model_or_runner(seq_len_in_pa, seq_len_out_pa, warmup_bs=self.warmup_bs)

                            try:
                                self.pa_runner.warm_up()
                            except Exception as e:
                                self.error_catched = 1
                                self.error_message = str(e)
                                stack_trace = traceback.format_exc()
                                self.logger.error("Error caught during Warm-up: %s", self.error_message)
                                if isinstance(e, RuntimeError):
                                    self.logger.error("Warm-up failed.")
                                    self.logger.error(
                                        "If due to out of memory error, try to remove the largest memory-consuming "
                                        "case_pair: [%d, %d] and batch_size: %d combination." \
                                        % (self.warmup_seq_in, self.warmup_seq_out, self.warmup_bs)
                                    )
                                raise e

                        if self.test_mode == "performance_single":
                            responses, token_nums, e2e_time = self.pa_runner.infer(
                                self.input_text_or_file, self.batch_size,
                                seq_len_out, True, self.is_chat_model)
                            if self.rank == 0:
                                for i, response in enumerate(responses):
                                    length = len(self.input_text_or_file)
                                    inputs = self.input_text_or_file
                                    if i < length:
                                        self.logger.info('Question[%d]: %s', i, inputs[i])
                                    self.logger.info('Answer[%d]: %s', i, response)
                                    self.logger.info('Generate[%d] token num: %s', i, token_nums[i])
                        else:
                            input_ids = [torch.randint(0, self.pa_runner.model.config.vocab_size, [seq_len_in],
                                                    dtype=torch.int64) for _ in range(self.batch_size)]
                            if self.lora_adapter:
                                input_ids = self.__prepare_inputs_with_lora(0, input_ids)
                            _, _, e2e_time = self.pa_runner.infer(input_ids, self.batch_size,
                                                                seq_len_out, True, self.is_chat_model)
                        passed_cases += 1
                        self.logger.info(
                            "Inference completed for batch_size: %d, seq_len_in: %d, seq_len_out: %d, e2e_time: %f",
                            self.batch_size, seq_len_in, seq_len_out, e2e_time
                        )
                    except Exception as e:
                        self.error_catched = 1
                        self.error_message = str(e)
                        stack_trace = traceback.format_exc()
                        self.logger.error("Error caught during inference: %s", self.error_message)
                        self.logger.error("Stack trace:\n%s", stack_trace)
                        self.logger.info("Total passed cases: %d", passed_cases)
                        self.logger.info(
                            "Failed case details: batch_size: %d, seq_len_in: %d, seq_len_out: %d",
                            self.batch_size, seq_len_in, seq_len_out
                        )

                if self.rank == 0:
                    if not self.error_catched:
                        if self.model_type == "fa" and self.hardware_type == "GPU":
                            first_token_time_tensor = torch.load(
                                f"{self.data_dir}/first_token_time.pth",
                                weights_only=True
                            ).cpu()
                            first_token_time = first_token_time_tensor.item()
                            non_first_token_time_tensor = torch.load(
                                f"{self.data_dir}/non_first_token_time.pth", 
                                weights_only=True
                            ).cpu()
                            non_first_token_time = non_first_token_time_tensor.item() / (seq_len_out - 1)
                        else:
                            if self.model_type == "pa":
                                benchmark_csv = os.path.join(self.script_path, "../benchmark.csv")
                                with safe_open(benchmark_csv, newline='') as csvfile:
                                    csv_reader = csv.reader(csvfile)
                                    next(csv_reader)
                                    second_row = next(csv_reader)
                                    first_token_time = float(second_row[4]) / 1000
                                    non_first_token_time = float(second_row[5]) / 1000

                                try:
                                    non_first_token_throughput = self.batch_size / non_first_token_time
                                except ZeroDivisionError:
                                    non_first_token_throughput = 0
                                non_first_token_throughput_total += non_first_token_throughput
                                e2e_throughput = self.batch_size * seq_len_out / e2e_time
                                e2e_throughput_total += e2e_throughput

                                self.logger.info(
                                    "batch: %d, seq_len_in: %d, seq_len_out: %d, total_time: %f,"
                                    "first_token_time: %f, non_first_token_time: %f, "
                                    "non_first_token_throughput: %f, e2e_time: %f, e2e_throughput: %f", 
                                    self.batch_size, seq_len_in, seq_len_out, e2e_time, 
                                    first_token_time * 1000, non_first_token_time * 1000, 
                                    non_first_token_throughput, e2e_time, e2e_throughput
                                )
                                csv_results.append(
                                    [str(self.model_name).ljust(15), str(self.batch_size).ljust(15), 
                                    str(seq_len_in).ljust(15),
                                    str(seq_len_out).ljust(15),
                                    str(round(e2e_time, 10)).ljust(15), 
                                    str(round(first_token_time * 1000, 10)).ljust(25),
                                    str(round(non_first_token_time * 1000, 10)).ljust(25),
                                    str(round(non_first_token_throughput, 10)).ljust(36),
                                    str(round(e2e_throughput, 10)).ljust(25)])
                                if (self.test_mode == "performance_maxbs" and 
                                    non_first_token_time * 1000 > self.time_limit):
                                    self.satisfy_time_limit = 0
                    else:
                        csv_results.append(
                            [str(self.model_name).ljust(15), str(self.batch_size).ljust(15), str(seq_len_in).ljust(15),
                            str(seq_len_out).ljust(15), self.error_message.ljust(141)])
                
                if self.error_catched:
                    break

            if self.rank == 0 and self.model_type == "pa":
                try:
                    non_first_token_throughput_average = non_first_token_throughput_total / passed_cases
                    e2e_throughput_average = e2e_throughput_total / passed_cases
                except ZeroDivisionError:
                    non_first_token_throughput_average = 0.0
                    e2e_throughput_average = 0.0
                    self.logger.info("all cases failed")
                self.logger.info(
                    "batch: %d, non_first_token_throughput_total: %f "
                    "non_first_token_throughput_average: %f, "
                    "e2e_throughput_total: %f, e2e_throughput_average: %f",
                    self.batch_size, non_first_token_throughput_total, 
                    non_first_token_throughput_average, 
                    e2e_throughput_total, e2e_throughput_average)
                csv_results[-1].extend(
                    [str(round(non_first_token_throughput_average, 10)).ljust(45),
                     str(round(e2e_throughput_average, 10)).ljust(35)])
                if not os.path.exists(self.csv_formatted_path):
                    self.logger.warning("performance result csv formatted file not exist, skip recording results")
                    raise RuntimeError("csv result formatted file not exist")
                with safe_open(self.csv_formatted_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter='|')
                    for csv_result in csv_results:
                        csv_writer.writerow(csv_result)

                csv_results.insert(0, ["Model", "Batchsize", "In_seq", "Out_seq", "Total time(s)", 
                                       "First token time(ms)", "Non-first token time(ms)", 
                                       "Non-first token Throughput(Tokens/s)", "Throughput(Tokens/s)", 
                                       "Non-first token Throughput Average(Tokens/s)",
                                      "E2E Throughput Average(Tokens/s)"])
                df = pd.DataFrame(csv_results)
                df.to_csv(self.csv_path, index=False, header=False)

                self.logger.info("%s batch %d result saved in %s", 
                                 self.model_name, self.batch_size, self.csv_path)
                self.logger.info("%s batch %d formatted result saved in %s", 
                                 self.model_name, self.batch_size, self.csv_formatted_path)
        
        warmup()
        run_performance_test()
        self.logger.info("performance test end")

    def __run_precision(self):
        dataset_eval_func_table = {
            "BoolQ": self.__run_full_dataset_boolq,
            "CEval": self.__run_full_dataset_ceval_or_mmlu,
            "CMMLU": self.__run_full_dataset_cmmlu,
            "GSM8K": self.__run_full_dataset_gsm8k,
            "HumanEval": self.__run_full_dataset_humaneval,
            "HumanEval_X": self.__run_full_dataset_humaneval_x,
            "LongBench": self.__run_full_dataset_longbench,
            "LongBench-E": self.__run_full_dataset_longbench,
            "MMLU": self.__run_full_dataset_ceval_or_mmlu,
            "NeedleBench": self.__run_full_dataset_needlebench,
            "TruthfulQA": self.__run_full_dataset_truthfulqa
        }
        self.logger.info("precision test start")
        if self.test_mode == "precision_single":
            self.__run_precision_single()
        elif self.test_mode == "full" or self.test_mode == "lite":
            seq_len_in = 3072
            seq_len_out = 512
            if "LongBench" in self.dataset_name:
                seq_len_in = 76839
                seq_len_out = 256
            elif "NeedleBench" in self.dataset_name:
                seq_len_in = int(self.context_length[:-1]) * 1024
                seq_len_out = 256
            self.__get_model_or_runner(seq_len_in, seq_len_out)
            if self.hardware_type == "NPU":
                if self.model_type == "fa":
                    self.fa_runner.warm_up()
                else:
                    self.pa_runner.warm_up()
            self.csv_debug = {
                KEY: [],
                QUERIES: [],
                INPUT_TOKEN_IDS: [],
                OUTPUT_TOKEN_IDS: [], 
                TEST_RESULT: [],
                GOLDEN_RESULT: [],
                PASS: []
            }
            self.dataset_path = os.path.join(os.path.dirname(self.script_path), "data", self.dataset_name.lower())
            if self.test_mode == "lite":
                self.dataset_path += "_lite"
            dataset_eval_func = dataset_eval_func_table.get(self.dataset_name)
            if dataset_eval_func:
                dataset_eval_func()
            else:
                self.logger.error("%s not support", self.dataset_name)
                raise RuntimeError(f"{self.test_mode} not support")
        else:
            self.logger.error("%s not support", self.test_mode)
            raise RuntimeError(f"{self.test_mode} not support")
        self.logger.info("precision test end")

    def __run_precision_single(self):
        for seq_len_in, seq_len_out in self.local_case_pair:
            self.__get_model_or_runner(seq_len_in, seq_len_out)
            if self.model_type == "fa":
                if self.is_chat_model:
                    input_token_ids = self.__construct_inputids(self.input_text_or_file, self.is_chat_model)
                    self.input_text_or_file = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                            for ids in input_token_ids]
                inputs = self.tokenizer(self.input_text_or_file * self.batch_size, return_tensors=PT,
                    padding='max_length', max_length=seq_len_in)

                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                
                with torch.no_grad():
                    generate_ids = self.model.generate(inputs=input_ids,
                                                        attention_mask=attention_mask,
                                                        min_new_tokens=seq_len_out,
                                                        max_new_tokens=seq_len_out
                                                        )
                    try:
                        responses = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                    except AttributeError:
                        responses = [
                            self.tokenizer.decode(output)
                            for output in generate_ids[:, inputs["input_ids"].size(1):].tolist()
                        ]
            else:
                responses, token_nums, _ = self.pa_runner.infer(
                    self.input_text_or_file, 
                    self.batch_size, 
                    seq_len_out, 
                    False, 
                    self.is_chat_model
                )
            if self.rank == 0:
                for i, response in enumerate(responses):
                    length = len(self.input_text_or_file)
                    inputs = self.input_text_or_file
                    if i < length:
                        self.logger.info('Question[%d]: %s', i, inputs[i])
                    self.logger.info('Answer[%d]: %s', i, response)
                    if self.model_type == "pa":
                        self.logger.info('Generate[%d] token num: %s', i, token_nums[i])

    def __run_full_dataset_ceval_or_mmlu(self):
        if self.shot == 0:
            self.__run_full_dataset_ceval_or_mmlu_0_shot()
        else:
            self.__run_full_dataset_ceval_or_mmlu_few_shots()
    
    def __run_full_dataset_ceval_or_mmlu_0_shot(self):

        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            with safe_open(subject_mapping_path) as f:
                subject_mapping = json.load(f)
            return subject_mapping
        
        def load_csv_by_task_name(task_name, dataset_path):
            if self.dataset_name == "CEval":
                val_df = pd.read_csv(os.path.join(dataset_path, "val", task_name + "_val.csv"), header=None)
                val_df = val_df.iloc[1:, 1:]
            else:
                val_df = pd.read_csv(os.path.join(dataset_path, "test", task_name + "_test.csv"), header=None)
            return val_df
        
        def format_example(name, df, idx):
            question = df.iloc[idx, 0]
            option_a = df.iloc[idx, 1]
            option_b = df.iloc[idx, 2]
            option_c = df.iloc[idx, 3]
            option_d = df.iloc[idx, 4]
            if self.dataset_name == "CEval":
                prompt = (
                    f"\n以下是中国关于{name}考试的单项选择题，请选出其中的正确答案。"
                    f"\n{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n"
                    f"让我们一步一步思考。答案: "
                )
            else:
                prompt = (
                    "\nThere is a single choice question about {}. Answer the question by replying A, B, C or D.\n"
                    "Q: {}\nA. {}\nB. {}\nC. {}\nD. {}\nLet's think step by step. A: ".format(
                        name.replace("_", " "), 
                        question, 
                        option_a, 
                        option_b, 
                        option_c, 
                        option_d
                    )
                )
            return prompt

        status_correct = "Correct"
        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if self.__get_rank() == 0:
            is_result = True

        subject_mapping = get_subject_mapping()
        if self.dataset_name == "MMLU":
            subject_mapping = subject_mapping["mmlu_all_sets"]
        index = 1
        for task_name in tqdm(subject_mapping):
            self.logger.info("dataset %d start, task name: %s", index, task_name)
            val_df = load_csv_by_task_name(task_name, self.dataset_path)
            correct = 0
            task_len = val_df.shape[0]
            for i in range(math.ceil(task_len / self.batch_size)):
                answers = []
                answer_results = []
                try:
                    q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                    name = subject_mapping[task_name][1] if self.dataset_name == "CEval" else task_name
                    prompt = [format_example(name, val_df, i * self.batch_size + j) for j in range(q_num)]
                    labels = [val_df.iloc[i * self.batch_size + j, val_df.shape[1] - 1] for j in range(q_num)]
                    prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt]

                    if is_result:
                        for idx in range(q_num):
                            self.csv_debug.get(KEY).append(f"{task_name}_{i * self.batch_size + idx}")
                            self.csv_debug.get(QUERIES).append(prompts[idx])

                    if self.model_type == "fa":
                        if self.hardware_type == "NPU":
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            generate_texts, _, _ = self.fa_runner.infer(prompts)
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            
                            for idx, generate_text in enumerate(generate_texts):
                                if is_result:
                                    self.logger.debug('Question[%d]: %s', i * self.batch_size + idx, prompts[idx])
                                    self.logger.debug('Answer[%d]: %s', i * self.batch_size + idx, generate_text)
                            answers = None

                            if len(generate_texts) > 0:
                                answers = generate_texts
                        else:
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(prompts, self.is_chat_model)
                                prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            inputs = self.tokenizer(prompts, padding=True, return_tensors=PT, truncation=True)
                            for idx in range(q_num):
                                self.csv_debug.get(INPUT_TOKEN_IDS).append(inputs.input_ids[idx].tolist())
                            inputs = inputs.to(0)
                            tokenizer_out_ids = inputs.input_ids.to(0)
                            attention_mask = inputs.attention_mask.to(0)
                            outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                                do_sample=False, max_new_tokens=256)
                            answers = []
                            for idx, output in enumerate(outputs.tolist()):
                                output = output[len(inputs["input_ids"][idx]):]
                                self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output)
                                answers.append(self.tokenizer.decode(output))
                    else:
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                        os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                        local_batch_size = self.batch_size
                        if len(prompts) == 1:
                            local_batch_size = 1
                        generate_texts, token_nums, _ = self.pa_runner.infer(
                            prompts, 
                            local_batch_size, 
                            256, 
                            False, 
                            self.is_chat_model
                        )
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"

                        for idx, generate_text in enumerate(generate_texts):
                            if is_result:
                                self.logger.debug('Question[%d]: %s', i * self.batch_size + idx, prompts[idx])
                                self.logger.debug('Answer[%d]: %s', i * self.batch_size + idx, generate_text)
                                self.logger.debug('Generate[%d] token num: %d', 
                                                  i * self.batch_size + idx, token_nums[idx])

                        answers = generate_texts

                        if len(generate_texts) > 0:
                            answers = generate_texts

                        if is_result:
                            for idx in range(q_num):
                                input_token_ids = torch.load(
                                    os.path.join(self.data_dir, f'input_ids_{idx}.pth'),
                                    weights_only=True
                                )
                                self.csv_debug.get(INPUT_TOKEN_IDS).append(input_token_ids.tolist())
                                with safe_open(os.path.join(self.data_dir, f"output_ids_{idx}.txt"), 'r') as f:
                                    output_token_ids = list(map(int, f.read().split()))
                                self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output_token_ids)
                    
                    if answers is None:
                        answers = []
                    
                    answer_results = [ModelTest._postprocess(answer, "ABCD") for answer in answers]
                    is_correct = [
                        status_correct 
                        if answer_result == label 
                        else "Wrong" for answer_result, label in zip(answer_results, labels)
                    ]
                    
                    correct += is_correct.count(status_correct)
                    for idx, is_pass in enumerate(is_correct):
                        self.csv_debug.get(GOLDEN_RESULT).append(labels[idx])
                        self.csv_debug.get(TEST_RESULT).append(answer_results[idx])
                        self.csv_debug.get(PASS).append(is_pass)
                        if is_result and is_pass != status_correct:
                            self.logger.debug(">>>原始题目 is : %s", prompts[idx])
                            self.logger.debug(">>>推理结果 is : %s", answer_results[idx])
                            self.logger.debug(">>>真实结果 is : %s", labels[idx])
                except Exception as e:
                    self.error_catched += q_num
                    self.logger.error("Error occurred in batch %d: %s", i, str(e), exc_info=True)

            if is_result:        
                result = [task_name, correct / task_len, correct, task_len]
                self.logger.info("dataset %d finish, result: %s", index, result)
                result_total.append(result)
                correct_total += correct
                sum_total += task_len
            index += 1

        if is_result:
            total = ["total", correct_total / sum_total, correct_total, sum_total]
            self.error_rate = self.error_catched / len(subject_mapping)
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.logger.debug("total result: %s", total)
            result_total.insert(0, total)
            if self.model_type == "pa":
                self.__save_debug()
                self.__save_result(result_total)
    
    def __run_full_dataset_ceval_or_mmlu_few_shots(self):
        choices = ["A", "B", "C", "D"]
        test_set = {"CEval": "val", "MMLU": "test"}
        status_correct = "Correct"

        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            with safe_open(subject_mapping_path) as f:
                subject_mapping = json.load(f)
            return subject_mapping if self.dataset_name == "CEval" else subject_mapping["mmlu_all_sets"]
        
        def load_csv_by_task_name(task_name, dataset_path):
            row_begin_idx = 0 if self.dataset_name == "MMLU" else 1
            col_begin_idx = 0 if self.dataset_name == "MMLU" else 1
            ori_dev_df = pd.read_csv(os.path.join(dataset_path, "dev", task_name + "_dev.csv"), header=None)
            ori_val_df = pd.read_csv(os.path.join(dataset_path, test_set.get(self.dataset_name), 
                                                  f"{task_name}_{test_set.get(self.dataset_name)}.csv"), header=None)

            dev_df = ori_dev_df.iloc[row_begin_idx:row_begin_idx + self.shot, col_begin_idx:]
            val_df = ori_val_df.iloc[row_begin_idx:, col_begin_idx:]
            return dev_df, val_df
        
        def format_subject(subject):
            subject_parts = subject.split("_")
            formatted_subjects = ""
            for entry in subject_parts:
                formatted_subjects += " " + entry    
            return formatted_subjects
            
        def format_example(df, idx, include_answer=True):
            prompt = df.iloc[idx, 0]
            k = len(choices)
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
            prompt += "\nAnswer:"
            if self.is_instruct_model:
                user_conversation = create_single_conversation("user", prompt)
                conversations = [user_conversation]
                if include_answer:
                    prompt += " {}\n\n".format(df.iloc[idx, k + 1])
                    assistant_conversation = create_single_conversation("assistant", df.iloc[idx, k + 1])
                    conversations.append(assistant_conversation)
                return conversations
            else:
                if include_answer:
                    prompt += " {}\n\n".format(df.iloc[idx, k + 1])
                return prompt
        
        def gen_prompt(train_df, subject, k=-1):
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(subject)
            )
            if k == -1:
                k = train_df.shape[0]
            if self.is_instruct_model:
                system_conversation = create_single_conversation("system", prompt)
                conversation_list = [system_conversation]
                for i in range(k):
                    conversation_list += format_example(train_df, i)
                return conversation_list
            else:
                for i in range(k):
                    prompt += format_example(train_df, i)
                return prompt

        def create_single_conversation(role, content):
            conversation_dict = dict()
            conversation_dict["role"] = role
            conversation_dict["content"] = content.encode().decode(encoding="utf8")
            return conversation_dict


        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if self.__get_rank() == 0:
            is_result = True

        subject_mapping = get_subject_mapping()
        index = 1
        for task_name in tqdm(subject_mapping):
            self.logger.info("dataset %d start, task name: %s", index, task_name)
            dev_df, val_df = load_csv_by_task_name(task_name, self.dataset_path)
            correct = 0
            task_len = val_df.shape[0]
            for i in range(math.ceil(task_len / self.batch_size)):
                try:
                    q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                    prompt_ends = [format_example(val_df, i * self.batch_size + j, include_answer=False) 
                                for j in range(q_num)]
                    train_prompts = [gen_prompt(dev_df, task_name, self.shot)] * q_num
                    prompt = [t + p for t, p in zip(train_prompts, prompt_ends)]
                    labels = [val_df.iloc[i * self.batch_size + j, val_df.shape[1] - 1] for j in range(q_num)]
                    prompts = [prpt.encode().decode(encoding="utf8")
                               for prpt in prompt] if not self.is_instruct_model else prompt

                    if is_result:
                        for idx in range(q_num):
                            self.csv_debug.get(KEY).append(f"{task_name}_{i * self.batch_size + idx}")
                            self.csv_debug.get(QUERIES).append(prompts[idx])

                    if self.model_type == "fa":
                        if self.hardware_type == "NPU":
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            generate_texts, _, _ = self.fa_runner.infer(prompts)
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            for idx, generate_text in enumerate(generate_texts):
                                if is_result:
                                    self.logger.debug('Question[%d]: %s', i * self.batch_size + idx, prompts[idx])
                                    self.logger.debug('Answer[%d]: %s', i * self.batch_size + idx, generate_text)
                            answers = None

                            if len(generate_texts) > 0:        
                                answers = generate_texts
                        else:
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(prompts, self.is_chat_model)
                                prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            inputs = self.tokenizer(prompts, padding=True, return_tensors=PT, truncation=True)
                            for idx in range(q_num):
                                self.csv_debug.get(INPUT_TOKEN_IDS).append(inputs.input_ids[idx].tolist())
                            inputs = inputs.to(0)
                            tokenizer_out_ids = inputs.input_ids.to(0)
                            attention_mask = inputs.attention_mask.to(0)
                            outputs = self.model.generate(
                                inputs=tokenizer_out_ids, 
                                attention_mask=attention_mask, 
                                do_sample=False, 
                                max_new_tokens=20
                            )
                            answers = []
                            for idx in range(len(outputs)):
                                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                                self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output)
                                response = self.tokenizer.decode(output)
                                answers.append(response)
                    else:
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                        os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                        local_batch_size = self.batch_size
                        if len(prompts) == 1:
                            local_batch_size = 1
                        truncation = False
                        if self.model_name == "llama_33b":
                            truncation = True
                        if self.lora_adapter:
                            infer_input = self.__prepare_inputs_with_lora(i * self.batch_size, prompts)
                        else:
                            infer_input = prompts
                        generate_texts, token_nums, _ = self.pa_runner.infer(
                                infer_input,
                                local_batch_size,
                                20,
                                False,
                                self.is_chat_model,
                                truncation=truncation
                            )
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"

                        for idx, generate_text in enumerate(generate_texts):
                            if is_result:
                                self.logger.debug('Question[%d]: %s', i * self.batch_size + idx, prompts[idx])
                                self.logger.debug('Answer[%d]: %s', i * self.batch_size + idx, generate_text)
                                self.logger.debug('Generate[%d] token num: %d', 
                                                  i * self.batch_size + idx, token_nums[idx])

                        answers = None

                        if len(generate_texts) > 0:
                            answers = generate_texts

                        if is_result:
                            for idx in range(q_num):
                                input_token_ids = torch.load(
                                    os.path.join(self.data_dir, f'input_ids_{idx}.pth'),
                                    weights_only=True
                                )
                                self.csv_debug.get(INPUT_TOKEN_IDS).append(input_token_ids.tolist())
                                with safe_open(os.path.join(self.data_dir, f"output_ids_{idx}.txt"), 'r') as f:
                                    output_token_ids = list(map(int, f.read().split()))
                                self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output_token_ids)

                    answer_results = [answer.lstrip()[0] if answer else "-1" for answer in answers]
                    is_correct = [status_correct if answer_result == label else "Wrong" 
                                for answer_result, label in zip(answer_results, labels)]
                    
                    correct += is_correct.count(status_correct)
                    for idx, is_pass in enumerate(is_correct):
                        self.csv_debug.get(GOLDEN_RESULT).append(labels[idx])
                        self.csv_debug.get(TEST_RESULT).append(answer_results[idx])
                        self.csv_debug.get(PASS).append(is_pass)
                        if is_result and is_pass != status_correct:
                            self.logger.debug(">>>原始题目 is : %s", prompts[idx])
                            self.logger.debug(">>>推理结果 is : %s", answer_results[idx])
                            self.logger.debug(">>>真实结果 is : %s", labels[idx])
                except Exception as e:
                    self.error_catched += q_num
                    self.logger.error("Error occurred %s", str(e), exc_info=True)

            if is_result:        
                result = [task_name, correct / task_len, correct, task_len]
                self.logger.info("dataset %d finish, result: %s", index, result)
                result_total.append(result)
                correct_total += correct
                sum_total += task_len
            index += 1

        if is_result:
            total = ["total", correct_total / sum_total, correct_total, sum_total]
            self.error_rate = self.error_catched / len(subject_mapping)
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.logger.debug("total result: %s", total)
            result_total.insert(0, total)
            if self.model_type == "pa" or self.hardware_type == "GPU":
                self.__save_debug()
                self.__save_result(result_total)

    def __run_full_dataset_cmmlu(self):
        choices = ["A", "B", "C", "D"]
        tokenizer = self.tokenizer if self.model_type == "fa" else self.pa_runner.tokenizer
        choice_ids = [tokenizer.convert_tokens_to_ids(choice) for choice in choices]

        is_result = False
        if self.__get_rank() == 0:
            is_result = True

        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            with safe_open(subject_mapping_path) as f:
                subject_mapping = json.load(f)
            return subject_mapping
        
        def format_example(df, idx, subject, include_answer=True, cot=False):
            prompt_start = "题目："
            prompt = prompt_start + df.iloc[idx, 0]
            k = df.shape[1] - 2
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])

            if cot:
                prompt += "\n逐步分析并给出答案选项。"
            else:
                prompt += "\n答案是："

            if include_answer:
                prompt += "{}\n\n".format(df.iloc[idx, k + 1])
            return prompt
        
        def gen_prompt(dev_df, subject, prompt_end, num_few_shot=0, tokenizer=None):
            max_length = 2048
            cot = False

            subject_name = name_en2zh.get(subject, "未知主题")
            if cot:
                prompt = "以下是关于{}的单项选择题，请分析并选出正确答案。\n\n".format(subject_name)
            else:
                prompt = "以下是关于{}的单项选择题，请直接给出正确答案的选项。\n\n".format(subject_name)

            if tokenizer is None:
                for i in range(num_few_shot):
                    example = format_example(dev_df, i, subject)
                    prompt += example
                return prompt + prompt_end

            start_end_token_len = len(tokenizer.encode(prompt) + tokenizer.encode(prompt_end))
            if start_end_token_len > max_length:
                return prompt_end

            prompt_list = []
            if num_few_shot > 0:
                for i in range(num_few_shot):
                    example = format_example(dev_df, i, subject)
                    prompt_list.append((example, tokenizer.encode(example)))

                while prompt_list and sum(len(e[1]) for e in prompt_list) >= max_length - start_end_token_len:
                    self.logger.warning("Warning: %d shot case exceeds max_input_length, remove 1 shot.", 
                                        len(prompt_list))
                    longest_length = max([len(e[1]) for e in prompt_list])
                    prompt_list = [e for e in prompt_list if len(e[1]) != longest_length]
                for p in prompt_list:
                    prompt += p[0]

            return prompt + prompt_end
        
        def softmax(x):
            z = x - max(x)
            numerator = np.exp(z)
            denominator = np.sum(numerator)
            softmax = numerator / denominator
            return softmax
        
        subject_mapping = get_subject_mapping()["cmmlu_all_sets"]
        dataset_index = 1
        sum_total = 0
        for task_name in tqdm(subject_mapping):
            self.logger.info("dataset %d start, task name: %s", dataset_index, task_name)
            out_file = os.path.join(self.debug_dir, f"results_{task_name}.csv")
            dev_df = pd.read_csv(os.path.join(self.dataset_path, "dev", task_name + ".csv"), header=0, index_col=0)
            test_df = pd.read_csv(os.path.join(self.dataset_path, "test", task_name + ".csv"), header=0, index_col=0)
            task_len = test_df.shape[0]

            cors = []
            all_conf = []
            all_preds = []
            all_time = []

            for i in range(math.ceil(task_len / self.batch_size)):
                try:
                    q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                    sum_total += q_num
                    prompt_ends = [
                        format_example(test_df, i * self.batch_size + j, task_name, include_answer=False) 
                        for j in range(q_num)
                        ]
                    prompts = [gen_prompt(dev_df=dev_df,
                                        subject=task_name,
                                        prompt_end=prompt_end,
                                        num_few_shot=self.shot,
                                        tokenizer=tokenizer) 
                                        for prompt_end in prompt_ends]
                    labels = [test_df.iloc[i * self.batch_size + j, test_df.shape[1] - 1] for j in range(q_num)]

                    if is_result:
                        for idx in range(q_num):
                            self.csv_debug.get(KEY).append(f"{task_name}_{i * self.batch_size + idx}")
                            self.csv_debug.get(QUERIES).append(prompts[idx])
                    
                    if self.model_type == "fa":
                        if self.is_chat_model:
                            input_token_ids = self.__construct_inputids(prompts, self.is_chat_model)
                            prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                    for ids in input_token_ids]
                        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors=PT)
                        for idx in range(q_num):
                            self.csv_debug.get(INPUT_TOKEN_IDS).append(inputs.input_ids[idx].tolist())
                        inputs = inputs.to(0)
                        if "token_type_ids" in inputs:
                            inputs.pop("token_type_ids")

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            last_token_logits = outputs.logits[:, -1, :]
                            output_token_ids = last_token_logits.argmax(dim=-1)
                            self.csv_debug.get(OUTPUT_TOKEN_IDS, []).extend(output_token_ids.tolist())
                    else:
                        os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
                        os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.data_dir
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                        os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                        local_batch_size = self.batch_size
                        if len(prompts) == 1:
                            local_batch_size = 1
                        _, _, e2e_time = self.pa_runner.infer(prompts, local_batch_size, 1, False, self.is_chat_model)
                        os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                        if is_result:
                            for idx in range(q_num):
                                input_token_ids = torch.load(
                                    os.path.join(self.data_dir, f'input_ids_{idx}.pth'),
                                    weights_only=True
                                )
                                self.csv_debug.get(INPUT_TOKEN_IDS, []).append(input_token_ids.tolist())
                                with safe_open(os.path.join(self.data_dir, f"output_ids_{idx}.txt"), 'r') as f:
                                    output_token_ids = list(map(int, f.read().split()))
                                self.csv_debug.get(OUTPUT_TOKEN_IDS, []).append(output_token_ids)
                            last_token_logits = torch.load(
                                os.path.join(self.data_dir, 'logits_0.pth'),
                                weights_only=True
                            )
                    
                    if is_result:
                        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
                        for idx, label in enumerate(labels):
                            conf = softmax(choice_logits[idx])[choices.index(label)]
                            pred = {0: "A", 1: "B", 2: "C", 3: "D"}.get(np.argmax(choice_logits[idx]), "Unknown")

                            all_preds += pred
                            all_conf.append(conf)
                            all_time.append(e2e_time)
                            cors.append(pred == label)
                            self.csv_debug.get(GOLDEN_RESULT, []).append(label)
                            self.csv_debug.get(TEST_RESULT, []).append(pred)
                            self.csv_debug.get(PASS, []).append(pred == label)
                except Exception as e:
                    self.error_catched += q_num
                    self.logger.error("Error occurred %s", str(e), exc_info=True)

            if is_result:
                acc = np.mean(cors)
                self.logger.debug("Average accuracy %.3f - %s", acc, task_name)
                test_df['prediction'] = all_preds
                test_df['e2e_time'] = all_time
                test_df.to_csv(out_file, header=None)
                
            dataset_index += 1

        if is_result:
            self.error_rate = self.error_catched / sum_total
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.__save_result("")
            self.__save_debug()

    def __run_full_dataset_needlebench(self):

        def trim_prediction(prediction, reference):
            l08 = int(0.8 * len(reference))
            l12 = int(1.2 * len(reference))
            trimmed_prediction = prediction[:l12]

            if len(trimmed_prediction) > l08 and \
            reference[-1] in trimmed_prediction[l08:]:
                end_pos = l08 + trimmed_prediction[l08:].index(reference[-1]) + 1
                trimmed_prediction = trimmed_prediction[:end_pos]

            return trimmed_prediction

        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)

            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        def score(prediction, gold):
            total_score = 0
            keyword = gold.split('*')[1]
            reference = gold.split('*')[0]
            raw_prediction = prediction
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)

            prediction = trim_prediction(prediction, reference)

            edit_distance = levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 1 - edit_distance / max_len if max_len != 0 else 1

            if keyword in raw_prediction:
                score = 1
            else:
                score = 0.2 * score

            detail = {
            'pred': prediction,
            'answer': reference,
            'edit_distance': edit_distance,
            'score': score
            }
            total_score += score
            result = {'score': total_score, 'detail': detail}
            return result
        is_result = False
        context_length = self.context_length
        if self.__get_rank() == 0:
            is_result = True
        context_length_test = {
            '4k':needlebench_single.test_single_4k,
            '8k':needlebench_single.test_single_8k,
            '32k':needlebench_single.test_single_32k,
            '128k':needlebench_single.test_single_128k,
            '256k':needlebench_single.test_single_256k,
            '200k':needlebench_single.test_single_200k,
            '1000k':needlebench_single.test_single_1000k
        }
        datasets = context_length_test.get(context_length)()
        correct_total = 0
        sum_total = 0
        result_total = []
        with torch.no_grad():
            for dataset in tqdm(datasets, desc='global'):
                correct = 0
                sample_count = len(dataset.get('data'))
                dataloader = torch.utils.data.DataLoader(dataset.get('data'), batch_size=self.batch_size)
                for batch in tqdm(dataloader):
                    queries = batch["prompt"]
                    if self.model_type == "fa":
                        if self.is_chat_model:
                            input_token_ids = self.__construct_inputids(queries, self.is_chat_model)
                            queries = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                    for ids in input_token_ids]
                        self.model.set_tokenizer(self.tokenizer)
                        sampling_params = SamplingParams(temperature=0, max_tokens=256)
                        output = self.model.generate(queries, sampling_params)
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                self.csv_debug.get(KEY).append(idx)
                                self.csv_debug.get(INPUT_TOKEN_IDS, []).append(output[0].prompt_token_ids)
                                self.csv_debug.get(OUTPUT_TOKEN_IDS, []).extend([output[0].outputs[0].token_ids])
                                response = output[0].outputs[0].text
                                acc = score(response, ans).get('score')
                                correct += acc
                                self.csv_debug.get(QUERIES).append(queries[idx])
                                self.csv_debug.get(GOLDEN_RESULT).append(ans)
                                self.csv_debug.get(TEST_RESULT).append(response)
                                self.csv_debug.get(PASS).append(acc)
                    else:
                        os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
                        os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.data_dir
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                        os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                        local_batch_size = self.batch_size
                        if len(queries) == 1:
                            local_batch_size = 1
                        generate_texts, _, _ = self.pa_runner.infer(queries, local_batch_size,
                            256, False, self.is_chat_model)
                        os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                self.csv_debug.get(KEY).append(idx)
                                input_token_ids = torch.load(
                                    os.path.join(self.data_dir, f'input_ids_{idx}.pth'),
                                    weights_only=True
                                )
                                self.csv_debug.get(INPUT_TOKEN_IDS).append(input_token_ids.tolist())
                                with safe_open(os.path.join(self.data_dir, f"output_ids_{idx}.txt"), 'r') as f:
                                    output_token_ids = list(map(int, f.read().split()))
                                self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output_token_ids)
                                response = generate_texts[idx]
                                self.csv_debug.get(QUERIES).append(queries[idx])
                                self.csv_debug.get(GOLDEN_RESULT).append(ans)
                                self.csv_debug.get(TEST_RESULT).append(response)
                                acc = score(response, ans).get('score')
                                correct += acc
                                self.csv_debug.get(PASS).append(acc)                                                                                                                                                           
                dataset_name = dataset.get('name')
                result = [dataset_name, correct / sample_count, correct, sample_count]
                self.logger.debug("result: %s", result)
                result_total.append(result)
                correct_total += correct
                sum_total += sample_count

            total = ["total", correct_total / sum_total, correct_total, sum_total]
            result_total.insert(0, total)
        if is_result:
            self.__save_debug()
            self.__save_result(result_total)

    def __run_full_dataset_gsm8k(self):
        def extract_answer(s):
            pattern = r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
            _pat_last_digit = re.compile(pattern)
            match = list(_pat_last_digit.finditer(s))
            if match:
                last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
            else:
                last_digit = None
            return last_digit

        def is_correct(completion, answer):
            gold = extract_answer(answer)
            if gold is None:
                return False

            def number_equal(answer, pred):
                if pred is None:
                    return False
                try:
                    answer_dec = Decimal(answer)
                    pred_dec = Decimal(pred)
                    return math.isclose(answer_dec, pred_dec, rel_tol=0, abs_tol=Decimal('1e-4'))
                except (InvalidOperation, ValueError, TypeError, SyntaxError) as e:
                    self.logger.error("Error evaluating expression: %s", str(e))
                    return False
                except OverflowError as e:
                    self.logger.error("OverflowError: %s", str(e))
                    return False

            return number_equal(gold, extract_answer(completion))

        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if self.__get_rank() == 0:
            is_result = True
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(),
                                        recursive=True), desc='global'):
                dataset = []
                with safe_open(entry, encoding='utf-8') as f:
                    for line in f:
                        dataset.append(json.loads(line))

                correct = 0
                sample_count = len(dataset)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for idxx, batch in enumerate(tqdm(dataloader)):
                    try:
                        texts = batch["question"]
                        queries = [query for query in texts]
                        
                        if self.model_type == "fa":
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(queries, self.is_chat_model)
                                queries = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            inputs = self.tokenizer(queries, padding=True, return_tensors=PT, truncation=True,
                                                        max_length=2048).to(self.model.device)
                            tokenizer_out_ids = inputs.input_ids.to(self.model.device)
                            attention_mask = inputs.attention_mask.to(self.model.device)
                            outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                        do_sample=False, max_new_tokens=512)
                            if is_result:
                                for idx, ans in enumerate(batch['answer']):
                                    self.csv_debug.get(KEY).append(idx)
                                    question = texts[idx]
                                    self.csv_debug.get(INPUT_TOKEN_IDS).append(inputs.input_ids[idx].tolist())
                                    output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                                    self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output)
                                    response = self.tokenizer.decode(output)
                                    self.csv_debug.get(QUERIES).append(question)
                                    self.csv_debug.get(GOLDEN_RESULT).append(ans)
                                    self.csv_debug.get(TEST_RESULT).append(response)
                                    acc = is_correct(response, ans)
                                    self.csv_debug.get(PASS).append(acc)
                                    if acc:
                                        correct += 1
                        else:
                            local_batch_size = self.batch_size
                            if len(queries) == 1:
                                local_batch_size = 1
                            if self.lora_adapter:
                                infer_input = self.__prepare_inputs_with_lora(idxx * self.batch_size, queries)
                            else:
                                infer_input = queries
                            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
                            os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.data_dir
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            generate_texts, _, _ = self.pa_runner.infer(infer_input, local_batch_size, \
                                                                        512, False, self.is_chat_model)
                            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            if is_result:
                                for idx, ans in enumerate(batch['answer']):
                                    self.csv_debug.get(KEY).append(idx)
                                    input_token_ids = torch.load(
                                        os.path.join(self.data_dir, f'input_ids_{idx}.pth'),
                                        weights_only=True
                                    )
                                    self.csv_debug.get(INPUT_TOKEN_IDS).append(input_token_ids.tolist())
                                    with open(os.path.join(self.data_dir, f"output_ids_{idx}.txt"), 'r') as f:
                                        output_token_ids = list(map(int, f.read().split()))
                                    self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output_token_ids)
                                    question = texts[idx]
                                    response = generate_texts[idx]
                                    self.csv_debug.get(QUERIES).append(question)
                                    self.csv_debug.get(GOLDEN_RESULT).append(ans)
                                    self.csv_debug.get(TEST_RESULT).append(response)
                                    acc = is_correct(response, ans)
                                    self.csv_debug.get(PASS).append(acc)
                                    if acc:
                                        correct += 1
                    except Exception as e:
                        self.error_catched += len(queries)
                        self.logger.error("Error occurred %s", str(e), exc_info=True)

                filename = os.path.basename(entry)
                result = [filename, correct / sample_count, correct, sample_count]
                self.logger.debug("result: %s", result)
                result_total.append(result)
                correct_total += correct
                sum_total += sample_count

                            
            total = ["total", correct_total / sum_total, correct_total, sum_total]
            result_total.insert(0, total)

        if is_result:
            self.error_rate = self.error_catched / sum_total
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.__save_debug()
            self.__save_result(result_total)

    def __run_full_dataset_truthfulqa(self):
        best_col = 'Best Answer'
        answer_col = 'Correct Answers'
        incorrect_col = 'Incorrect Answers'
        metric = "Metric"

        def run_answers():
            frame = pd.read_csv((Path(self.dataset_path) / "TruthfulQA.csv").as_posix())
            frame.dropna(axis=1, how='all', inplace=True)

            if tag not in frame.columns:
                frame[tag] = ''

            frame[tag].fillna('', inplace=True)
            frame[tag] = frame[tag].astype(str)

            num_rows = frame.shape[0]
            num_batches = math.ceil(num_rows / self.batch_size)

            seq_start = np.array(tokenizer('A:')['input_ids'])
            seq_end = np.array(tokenizer('Q:')['input_ids'])
            
            answer_error_catched = 0
            with torch.no_grad():
                for batch in tqdm(range(num_batches)):
                    try:
                        q_num = self.batch_size if (batch + 1) * self.batch_size <= num_rows \
                            else num_rows - self.batch_size * batch
                        idx_list = [i for i in range(batch * self.batch_size, batch * self.batch_size + q_num)]
                        prompt = [truthfulqa_eval.format_prompt(frame.loc[idx]) for idx in idx_list]

                        if self.model_type == "fa":
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(prompt, self.is_chat_model)
                                prompt = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            input_ids = tokenizer(prompt, padding=True, return_tensors=PT, truncation=True).input_ids
                            max_len = input_ids.shape[-1] + 50
                            input_ids = input_ids.to(0)
                            outputs = self.model.generate(input_ids, do_sample=False, max_length=max_len)
                            output_token_ids_list = [output[len(input_ids[idx]):] 
                                                    for idx, output in enumerate(outputs.tolist())] 
                            gen_arrs = np.array(output_token_ids_list)
                        else:
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            local_batch_size = self.batch_size
                            if len(prompt) == 1:
                                local_batch_size = 1
                            _, _, _ = self.pa_runner.infer(prompt, local_batch_size, 50, False, self.is_chat_model)
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            if is_result:
                                output_token_ids_list = []
                                for idx in range(q_num):
                                    input_token_ids = torch.load(
                                        os.path.join(self.data_dir, f'input_ids_{idx}.pth'),
                                        weights_only=True
                                    )
                                    self.csv_debug.get(INPUT_TOKEN_IDS).append(input_token_ids.tolist())
                                    with safe_open(os.path.join(self.data_dir, f"output_ids_{idx}.txt"), 'r') as f:
                                        output_token_ids = list(map(int, f.read().split()))
                                        output_token_ids_list.append(output_token_ids)
                                    self.csv_debug.get(OUTPUT_TOKEN_IDS).append(output_token_ids)
                                gen_arrs = np.array(output_token_ids_list)
                        
                        if is_result:
                            idx_start = [truthfulqa_eval.find_subsequence(gen_arr, seq_start, start=True)
                                        for gen_arr in gen_arrs]
                            idx_end = [truthfulqa_eval.find_subsequence(gen_arr, seq_end, start=False)
                                    for gen_arr in gen_arrs]

                            output_token_ids_list = [output_token_ids[idx_start[output_token_ids_idx]:
                                                                    idx_end[output_token_ids_idx]] 
                            for output_token_ids_idx, output_token_ids in enumerate(output_token_ids_list)]
                            output_strs = [tokenizer.decode(output_token_ids, skip_special_tokens=True)
                                        for output_token_ids in output_token_ids_list]
                            output_str = [output_str.strip() for output_str in output_strs]

                            for idx in idx_list:
                                frame.loc[idx, tag] = output_str[idx % self.batch_size]
                            truthfulqa_answers_path = os.path.join(self.data_dir, 'truthfulQA_answers.csv')
                            frame.to_csv(truthfulqa_answers_path, index=False, header=True)
                            self.logger.info("%s TruthfulQA answers saved to: %s", tag, truthfulqa_answers_path)
                    except Exception as e:
                        answer_error_catched += q_num
                        self.logger.error("Error occurred %s", str(e), exc_info=True)
            answer_error_rate = answer_error_catched / num_rows
            self.logger.info("Failed answer cases: %d", answer_error_catched)
            self.logger.info("Failed answer rates: %.4f", answer_error_rate)
            return frame
        
        def run_probs(frame):
            truthfulqa_eval.set_columns(tag, frame)
            probs_error_catched = 0
            with torch.no_grad():
                for idx in tqdm(frame.index):
                    try:
                        if pd.isnull(frame.loc[idx, incorrect_col]):
                            self.logger.warning("References missing for %s!", idx)
                            continue
                        if not frame.loc[idx, incorrect_col]:
                            self.logger.warning("References missing for %s!", idx)
                            continue

                        ref_best = truthfulqa_eval.format_best(frame.loc[idx, best_col])
                        ref_true = truthfulqa_eval.split_multi_answer(frame.loc[idx, answer_col])
                        ref_false = truthfulqa_eval.split_multi_answer(frame.loc[idx, incorrect_col])

                        input_prompt = truthfulqa_eval.format_prompt(frame.loc[idx])

                        scores_true = get_scores(input_prompt, frame, idx, ref_true)
                        scores_false = get_scores(input_prompt, frame, idx, ref_false)

                        scores_ref = {
                            "scores_true": scores_true,
                            "scores_false": scores_false,
                            "ref_true": ref_true,
                            "ref_best": ref_best
                        }
                        
                        if is_result:
                            frame = truthfulqa_eval.mc_calcs(tag, frame, idx, scores_ref)
                    except Exception as e:
                        probs_error_catched += 1
                        self.logger.error("Error occurred %s", str(e), exc_info=True)

            probs_error_rate = probs_error_catched / len(frame)
            self.logger.info("Failed probability cases: %d", probs_error_catched)
            self.logger.info("Failed probability rates: %.4f", probs_error_rate)
            return frame

        def get_scores(input_prompt, frame, idx, ref_answer):
            scores_answer = []
            for temp_ans in ref_answer:
                prompt = [truthfulqa_eval.format_prompt_with_answer_strings(frame.loc[idx, 'Question'], temp_ans)]
                input_ids = tokenizer(input_prompt, return_tensors=PT).input_ids
                prompt_ids = tokenizer(prompt, return_tensors=PT).input_ids
                if self.model_type == "fa":
                    if self.is_chat_model:
                        input_token_ids = self.__construct_inputids(prompt, self.is_chat_model)
                        prompt = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                for ids in input_token_ids]
                    input_ids = input_ids.to(0)
                    prompt_ids = prompt_ids.to(0)
                    logits = self.model(prompt_ids)[0].squeeze(0)
                    logits = logits[input_ids.shape[-1] - 1: -1, :]
                    
                else:
                    os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
                    os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.data_dir
                    local_batch_size = self.batch_size
                    if len(prompt) == 1:
                        local_batch_size = 1
                    _, _, _ = self.pa_runner.infer(prompt, local_batch_size, 1, False, self.is_chat_model)
                    os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
                    if is_result:
                        logits = torch.load(
                            os.path.join(self.data_dir, 'logits_0.pth'),
                            weights_only=True
                        )

                if is_result:
                    logits_softmax = F.log_softmax(logits.float(), dim=-1)
                    prompt_ids = prompt_ids[0, input_ids.shape[-1]:]
                    log_probs = logits_softmax[range(logits_softmax.shape[0]), prompt_ids.squeeze(0)]
                    log_probs = log_probs[3:]
                    scores_answer.append(log_probs.sum().item())

            return scores_answer

        is_result = False
        if self.__get_rank() == 0:
            is_result = True
        tokenizer = self.tokenizer if self.model_type == "fa" else self.pa_runner.tokenizer
        tag = self.model_name

        frame = run_answers()
        frame = run_probs(frame)
        if is_result:
            frame = truthfulqa_eval.run_bleu_and_rouge(self.model_name, frame)
        
            results = truthfulqa_eval.format_frame(frame)

            truthfulqa_full_scores_path = os.path.join(self.data_dir, 'truthfulQA_full_scores.csv')
            frame.to_csv(truthfulqa_full_scores_path, index=False, header=True)
            self.logger.info("%s TruthfulQA full scores saved to: %s", tag, truthfulqa_full_scores_path)

            results = results.mean(axis=0)
            results = results.reset_index().rename(columns={'level_0': 'Model',
                                                            'level_1': metric,
                                                            0: 'Value'})

            results = results[results[metric].isin(['MC1', 'MC2',
                                                    'bleu diff',
                                                    'rouge1 diff',
                                                    'BLEURT diff'])]
            
            results = pd.pivot_table(results, 'Value', 'Model', metric)

            results = results.rename(columns={'bleu diff': 'BLEU',
                                                'rouge1 diff': 'ROUGE',
                                                'BLEURT diff': 'BLEURT'})
    
            self.__save_result(results)

    def __run_full_dataset_boolq(self):
        sample_yes = "How can we learning machine learning: yes"
        sample_no = "How can we learning machine learning: no"
        if self.model_type == "fa":
            choice_tokens = [
                self.tokenizer(
                    [sample_yes], 
                    return_tensors=PT, 
                    max_length=2048, 
                    add_special_tokens=False
                ).input_ids[0, -1].item(), 
                self.tokenizer(
                    [sample_no], 
                    return_tensors=PT, 
                    max_length=2048, 
                    add_special_tokens=False
                ).input_ids[0, -1].item()
            ]
        else:
            choice_tokens = [
                self.pa_runner.tokenizer(
                    [sample_yes], 
                    return_tensors=PT, 
                    max_length=2048, 
                    add_special_tokens=False
                ).input_ids[0, -1].item(), 
                self.pa_runner.tokenizer(
                    [sample_no], 
                    return_tensors=PT, 
                    max_length=2048, 
                    add_special_tokens=False
                ).input_ids[0, -1].item()
            ]
        
        def build_prompt(title, text, passage):
            prompt = f"{title} -- {passage}\nQuestion: {text}?\nAnswer:"
            return prompt

        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if self.__get_rank() == 0:
            is_result = True
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(),
                                        recursive=True), desc='global'):
                dataset = []
                with safe_open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                correct = 0
                sample_count = len(dataset)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for idx, batch in enumerate(tqdm(dataloader)):
                    try:
                        q_num = (
                            self.batch_size 
                            if (idx + 1) * self.batch_size <= sample_count 
                            else sample_count - idx * self.batch_size
                        )
                        titles = batch["title"]
                        texts = batch["question"]
                        passages = batch["passage"]
                        queries = [build_prompt(title, query, passage) 
                                for title, query, passage in zip(titles, texts, passages)]
                        if is_result:
                            for i in range(q_num):
                                self.csv_debug.get(KEY, []).append(idx * self.batch_size + i)
                                self.csv_debug.get(QUERIES, []).append(queries[i])

                        if self.model_type == "fa":
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(queries, self.is_chat_model)
                                queries = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            inputs = self.tokenizer(queries, padding=True, return_tensors=PT, truncation=True)
                            for i in range(q_num):
                                self.csv_debug.get(INPUT_TOKEN_IDS, []).append(inputs.input_ids[i].tolist())
                            inputs = inputs.to(0)
                            outputs = self.model(**inputs)
                            logits = outputs.logits[:, -1, :]
                            output_token_ids = logits.argmax(dim=-1)
                            self.csv_debug.get(OUTPUT_TOKEN_IDS, []).extend(output_token_ids.tolist())
                            logits_softmax = F.log_softmax(logits.float(), dim=-1)
                        else:
                            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
                            os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.data_dir
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            local_batch_size = self.batch_size
                            if len(queries) == 1:
                                local_batch_size = 1
                            if self.lora_adapter:
                                infer_input = self.__prepare_inputs_with_lora(idx * self.batch_size, queries)
                            else:
                                infer_input = queries
                            _, _, _ = self.pa_runner.infer(infer_input, local_batch_size, 1, False, self.is_chat_model)
                            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            if is_result:
                                for i in range(q_num):
                                    input_token_ids = torch.load(
                                        os.path.join(self.data_dir, f'input_ids_{i}.pth'),
                                        weights_only=True
                                    )
                                    self.csv_debug.get(INPUT_TOKEN_IDS, []).append(input_token_ids.tolist())
                                    with safe_open(os.path.join(self.data_dir, f"output_ids_{i}.txt"), 'r') as f:
                                        output_token_ids = list(map(int, f.read().split()))
                                    self.csv_debug.get(OUTPUT_TOKEN_IDS, []).append(output_token_ids)
                                logits = torch.load(
                                    os.path.join(self.data_dir, 'logits_0.pth'),
                                    weights_only=True
                                )
                                logits_softmax = F.log_softmax(logits.float(), dim=-1)
                        
                        if is_result:
                            logits_softmax = logits_softmax[:, choice_tokens]
                            for inner_idx, ans in enumerate(batch['answer']):
                                choice = (logits_softmax[inner_idx, 0] > logits_softmax[inner_idx, 1]).cpu()
                                acc = choice == ans
                                self.csv_debug.get(GOLDEN_RESULT, []).append(ans.item())
                                self.csv_debug.get(TEST_RESULT, []).append(choice.item())
                                self.csv_debug.get(PASS, []).append(acc.item())
                                if acc:
                                    correct += 1

                    except Exception as e:
                        self.error_catched += 1
                        self.logger.error("Error occurred %s", str(e), exc_info=True)

                if is_result:
                    filename = os.path.basename(entry)
                    result = [filename, correct / sample_count, correct, sample_count]
                    self.logger.info("result: %s", result)
                    result_total.append(result)
                    correct_total += correct
                    sum_total += sample_count
                
            if is_result:
                total = ["total", correct_total / sum_total, correct_total, sum_total]
                result_total.insert(0, total)
        if is_result:
            self.error_rate = self.error_catched / sample_count
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.__save_debug()
            self.__save_result(result_total)

    def __run_full_dataset_humaneval(self):
        
        def cleanup_code(code: str) -> str:
            code_splits = code.split("\n")
            is_empty_line = False
            ind_empty_line = None
            for i, line in enumerate(code_splits):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    is_empty_line = True                    
                    ind_empty_line = i                    
                    break            
            if is_empty_line:
                code = "\n".join(code_splits[:ind_empty_line])
            else:
                end_words = ["\ndef", "\nclass", "\n#", "\nassert", '\n"""', "\nprint", "\nif", "\n\n\n"]
                for w in end_words:
                    if w in code:
                        code = code[:code.rfind(w)]
            return code

        is_result = False
        if self.__get_rank() == 0:
            is_result = True
        self.csv_debug[CLEANED_UP_RESULTS] = self.csv_debug.pop(TEST_RESULT)
        self.csv_debug["test_cases"] = self.csv_debug.pop(GOLDEN_RESULT)
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(),
                                        recursive=True), desc='global'):
                dataset = []
                with safe_open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                samples = []
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for idx, batch in enumerate(tqdm(dataloader)):
                    try:
                        q_num = (
                            self.batch_size 
                            if (idx + 1) * self.batch_size <= len(dataset) 
                            else len(dataset) - idx * self.batch_size
                        )
                        task_ids = [task_id.split('/')[1] for task_id in batch[TASK_ID]]
                        queries = [prompt.strip() for prompt in batch[PROMPT]]
                        test_cases = [test_case for test_case in batch["test"]]
                        if is_result:
                            for i in range(q_num):
                                self.csv_debug.get(KEY, []).append(task_ids[i])
                                self.csv_debug.get(QUERIES, []).append(queries[i])
                                self.csv_debug.get("test_cases", []).append(test_cases[i])
                        if self.model_type == "fa":
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(queries, self.is_chat_model)
                                queries = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            inputs = self.tokenizer(queries, padding=True, return_tensors=PT, truncation=True)
                            for i in range(q_num):
                                self.csv_debug.get(INPUT_TOKEN_IDS, []).append(inputs.input_ids[i].tolist())
                            inputs = inputs.to(0)
                            tokenizer_out_ids = inputs.input_ids.to(0)
                            attention_mask = inputs.attention_mask.to(0)
                            outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                        do_sample=False, max_new_tokens=512)
                            if is_result:
                                for inner_idx, output in enumerate(outputs.tolist()):
                                    output = output[len(inputs["input_ids"][inner_idx]):]
                                    response = self.tokenizer.decode(output)
                                    response_cleaned_up = cleanup_code(response)
                                    self.csv_debug.get(OUTPUT_TOKEN_IDS, []).append(output)
                                    self.csv_debug.get(CLEANED_UP_RESULTS, []).append(response_cleaned_up)
                                    self.logger.info("response_cleaned_up: %s", response_cleaned_up)
                                    result = dict(
                                        task_id="HumanEval/" + task_ids[inner_idx],
                                        completion=response_cleaned_up,
                                    )
                                    samples += [result]
                        else:
                            local_batch_size = self.batch_size
                            if len(queries) == 1:
                                local_batch_size = 1
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            generate_text_list, _, _ = self.pa_runner.infer(
                                queries, 
                                local_batch_size, 
                                512, 
                                False, 
                                self.is_chat_model
                            )
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            generate_text_list = [cleanup_code(completion) for completion in generate_text_list]
                            if is_result:
                                self.logger.info("generate_text_list_cleaned_up: %s", generate_text_list)
                                for i in range(q_num):
                                    input_token_ids = torch.load(
                                        os.path.join(self.data_dir, f'input_ids_{i}.pth'),
                                        weights_only=True
                                    )
                                    self.csv_debug.get(INPUT_TOKEN_IDS, []).append(input_token_ids.tolist())
                                    with safe_open(os.path.join(self.data_dir, f"output_ids_{i}.txt"), 'r') as f:
                                        output_token_ids = list(map(int, f.read().split()))
                                    self.csv_debug.get(OUTPUT_TOKEN_IDS, []).append(output_token_ids)
                                    self.csv_debug.get(CLEANED_UP_RESULTS, []).append(generate_text_list[i])
                            for inner_idx, sample in enumerate(generate_text_list):
                                result = dict(
                                    task_id="HumanEval/" + task_ids[inner_idx],
                                    completion=sample,
                                )
                                samples += [result]
                    except Exception as e:
                        self.error_catched += 1
                        self.logger.error("Error occurred %s", str(e), exc_info=True)
                if is_result:
                    self.__save_result(samples)
                
        if is_result:
            results, passed_all = evaluate_functional_correctness(
                self.csv_path, 
                [1], 
                4, 
                3.0, 
                self.script_path + "/../data/humaneval/human-eval.jsonl"
            )
            self.csv_debug[PASS] = passed_all
            self.error_rate = self.error_catched / len(dataset)
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.__save_debug()
            self.logger.debug(results)

    def __run_full_dataset_humaneval_x(self):

        targeted_languages = ["cpp", "go", "java", "js", "python"]
        pass_at_ks = {lang: 0 for lang in targeted_languages}
        original_dataset_name = self.dataset_name
        original_csv_path = self.csv_path
        total_score = 0
        is_result = False
        if self.__get_rank() == 0:
            is_result = True
        for lang in targeted_languages:
            self.csv_debug = {
                KEY: [],
                QUERIES: [],
                INPUT_TOKEN_IDS: [],
                OUTPUT_TOKEN_IDS: [], 
                TEST_RESULT: [],
                GOLDEN_RESULT: [],
                PASS: []
            }
            self.csv_debug[CLEANED_UP_RESULTS] = self.csv_debug.pop(TEST_RESULT)
            self.csv_debug["test_cases"] = self.csv_debug.pop(GOLDEN_RESULT)
            self.dataset_name += f"_{lang}"
            os.environ['MODELTEST_DATASET_SPECIFIED'] = f"{self.dataset_name}"
            with torch.no_grad():
                dataset_path = os.path.join(self.dataset_path, lang, "data", f"data_{lang}_data_humaneval.jsonl")
                entries = humaneval_x_utils.read_dataset(dataset_path, dataset_type="humaneval")
                for entry in entries.values():
                    entry[PROMPT] = humaneval_x_utils.process_extra_prompt(entry[PROMPT], lang)
                dataset = humaneval_x_utils.HumanEvalXDataset(entries)

                samples = []
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for idx, batch in enumerate(tqdm(dataloader)):
                    try:
                        task_ids = batch[TASK_ID]
                        queries = batch[PROMPT]
                        test_cases = batch["test"]
                        q_num = self.batch_size if (idx + 1) * self.batch_size <= len(entries) \
                            else len(entries) - idx * self.batch_size
                        if is_result:
                            for i in range(q_num):
                                self.csv_debug[KEY].append(task_ids[i])
                                self.csv_debug[QUERIES].append(queries[i])
                                self.csv_debug["test_cases"].append(test_cases[i])
                        if self.model_type == "fa":
                            if self.is_chat_model:
                                input_token_ids = self.__construct_inputids(queries, self.is_chat_model)
                                queries = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                        for ids in input_token_ids]
                            inputs = self.tokenizer(queries, padding=True, return_tensors=PT, truncation=True)
                            for i in range(q_num):
                                self.csv_debug[INPUT_TOKEN_IDS].append(inputs.input_ids[i].tolist())
                            inputs = inputs.to(0)
                            tokenizer_out_ids = inputs.input_ids.to(0)
                            attention_mask = inputs.attention_mask.to(0)
                            stopping_criteria = StoppingCriteriaList()
                            stopping_criteria.append(
                                humaneval_x_utils.StoppingCriteriaWithHumanEvalX(
                                    lang=lang,
                                    original_input_len=inputs.input_ids.shape[1],
                                    tokenizer=self.tokenizer,
                                    ))
                            outputs = self.model.generate(
                                inputs=tokenizer_out_ids,
                                attention_mask=attention_mask,
                                do_sample=False,
                                stopping_criteria=stopping_criteria,
                                max_new_tokens=1024)
                            if is_result:
                                for output_idx, output in enumerate(outputs.tolist()):
                                    output = output[len(inputs["input_ids"][output_idx]):]
                                    response = self.tokenizer.decode(output)
                                    response_cleaned_up = humaneval_x_utils.cleanup_code(response, lang, 
                                                                                         self.dataset_name)
                                    self.csv_debug[OUTPUT_TOKEN_IDS].append(output)
                                    self.csv_debug[CLEANED_UP_RESULTS].append(response_cleaned_up)
                                    self.logger.info("response_cleaned_up: %s", response_cleaned_up)
                                    sample_format = {
                                        TASK_ID   : task_ids[output_idx],
                                        PROMPT    : queries[output_idx],
                                        "generation": response_cleaned_up,
                                    }
                                    samples += [sample_format]
                        else:
                            local_batch_size = self.batch_size
                            if len(queries) == 1:
                                local_batch_size = 1
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                            generate_text_list, _, _ = self.pa_runner.infer(
                                queries,
                                local_batch_size,
                                1024,
                                False,
                                self.is_chat_model
                            )
                            os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                            if is_result:
                                generate_text_list = [humaneval_x_utils.cleanup_code(
                                    completion,
                                    lang,
                                    self.dataset_name) for completion in generate_text_list]
                                self.logger.info("generate_text_list_cleaned_up: %s", generate_text_list)
                                for i in range(q_num):
                                    input_token_ids = torch.load(
                                        os.path.join(self.data_dir, f'input_ids_{i}.pth'),
                                        weights_only=True
                                    )
                                    self.csv_debug[INPUT_TOKEN_IDS].append(input_token_ids.tolist())
                                    with safe_open(os.path.join(self.data_dir, f"output_ids_{i}.txt"), 'r') as f:
                                        output_token_ids = list(map(int, f.read().split()))
                                    self.csv_debug[OUTPUT_TOKEN_IDS].append(output_token_ids)
                                    self.csv_debug[CLEANED_UP_RESULTS].append(generate_text_list[i])
                                for generate_text_list_idx, sample in enumerate(generate_text_list):
                                    sample_format = {
                                        TASK_ID   : task_ids[generate_text_list_idx],
                                        PROMPT    : queries[generate_text_list_idx],
                                        "generation": sample,
                                    }
                                    samples += [sample_format]
                    except Exception as e:
                        self.error_catched += 1
                        self.logger.error("Error occurred in batch %d: %s", idx, str(e), exc_info=True)
            if is_result:
                self.__save_result(samples)
                config = humaneval_x_eval.EvalConfig(
                    input_file=self.csv_path,
                    tmp_dir=self.data_dir,
                    n_workers=32,
                    timeout=500.0,
                    problem_file=dataset_path,
                    out_dir=self.result_dir,
                    k=[1],
                    go_dir=os.path.join(os.path.dirname(self.script_path), "dataset", "full", "HumanEval_X",
                                        "go", "evaluation")
                )
                pass_at_k, passed_all = humaneval_x_eval.evaluate_functional_correctness(config)
                pass_at_ks[lang] = pass_at_k
                total_score += pass_at_k.get("pass@1", 0)
                self.csv_debug[PASS] = passed_all
                self.__save_debug()
                self.logger.debug(pass_at_k)
            
            self.dataset_name = original_dataset_name
            self.csv_path = original_csv_path
            
        os.unsetenv('MODELTEST_DATASET_SPECIFIED')
        if is_result:
            self.error_rate = self.error_catched / len(dataloader)
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.logger.info("score map: %s", pass_at_ks)    
            self.logger.info("average score: %f", total_score / len(targeted_languages))

    def __run_full_dataset_longbench(self):
        answers = "answers"
        max_dataset_file_size = 30 * 1024 * 1024

        def load_dataset_by_task_name(task_name, suffix):
            file_path = os.path.join(self.dataset_path, "data", f"{task_name}{suffix}.jsonl")
            dataset = []
            with safe_open(file_path, 'r', encoding="utf-8", max_file_size=max_dataset_file_size) as f:
                for line in f:
                    data_line = json.loads(line)
                    dataset.append(data_line)
            return dataset
        
        def load_config():
            with safe_open(os.path.join(self.dataset_path, "./dataset2prompt.json"), "r") as file:
                task2prompt = json.load(file)
            with safe_open(os.path.join(self.dataset_path, "./dataset2maxlen.json"), "r") as file:
                task2maxgen = json.load(file)
            with safe_open(os.path.join(self.dataset_path, "./model2maxlen.json"), "r") as file:
                model2maxlen = json.load(file)
            return task2prompt, task2maxgen, model2maxlen

        def get_scores_by_task_name(task_name, task_result, use_longbench_e=False):
            predictions, golden_answers, lengths, all_classes = [], [], [], []
            for data in task_result:
                predictions.append(data["pred"])
                golden_answers.append(data[answers])
                all_classes = data[ALL_CLASSES]
                if LENGTH in data:
                    lengths.append(data[LENGTH])
            if use_longbench_e:
                score = scorer_e(task_name, predictions, golden_answers, lengths, all_classes)
            else:
                score = {"total_score": scorer(task_name, predictions, golden_answers, all_classes)}
            return {f"{task_name}": score}
        
        def get_final_scores(task_scores, task_nums, use_longbench_e=False):
            result_total = []
            score_total = []
            for task_name, res in task_scores.items():
                if use_longbench_e:
                    task_score_list = []
                    for _, score in res.items():
                        if not math.isnan(score):
                            task_score_list.append(score)
                    avg_task_scores = round(np.mean(task_score_list), 2)
                else:
                    avg_task_scores = res.get("total_score", 0.0)
                score_total.append(avg_task_scores)
                result_total.append([f"{task_name}", avg_task_scores, None, task_nums[task_name]])
            
            task_nums_list = np.array(list(task_nums.values()))
            score_total = round(np.average(score_total, weights=task_nums_list), 2)
            nums_total = np.sum(task_nums_list)
            result_total.insert(0, ["total", score_total, None, nums_total])
            return score_total, result_total

        self.csv_debug = {
            KEY: [],
            TEST_RESULT: [],
            GOLDEN_RESULT: [],
            LENGTH: [],
            ALL_CLASSES: [],
            OUTPUT_TOKEN_IDS: [],
            INPUT_TOKEN_IDS: []
        }
        use_longbench_e = self.dataset_name[-1] == "E"
        suffix = "_e" if use_longbench_e else ""
        if use_longbench_e:
            task_list = [
                "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
                "multi_news", "trec", "triviaqa", "samsum", "passage_count", 
                "passage_retrieval_en", "lcc", "repobench-p"
            ]
        else:
            task_list = [
                "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", 
                "2wikimqa", "musique", "dureader", "gov_report", "qmsum", 
                "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", 
                "repobench-p"
            ]
            if self.model_name == "baichuan2_13b":
                task_list = [
                    "narrativeqa", "hotpotqa", "triviaqa"
                ]
        
        result_total = []
        task_scores, task_nums = dict(), dict()
        is_result = self.__get_rank() == 0
        for idx, task_name in enumerate(tqdm(task_list, desc="global")):
            if is_result:
                self.logger.info("dataset %d start, task name: %s", idx + 1, task_name)
            dataset = load_dataset_by_task_name(task_name, suffix)
            task2prompt, task2maxgen, model2maxlen = load_config()
            prompts_pattern = task2prompt[task_name]
            max_new_tokens = task2maxgen[task_name]

            task_result = []
            for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f"{task_name}"):
                try:
                    prompts = prompts_pattern.format(**data)
                    if self.model_name in model2maxlen:
                        max_length = model2maxlen[self.model_name]
                        tokenized_prompt = self.tokenizer(prompts, truncation=False, return_tensors="pt").input_ids[0]
                        if len(tokenized_prompt) > max_length:
                            half = int(max_length / 2)
                            prompts = (self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                                       self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))
                    if self.model_type == "fa":
                        if self.is_chat_model:
                            input_token_ids = self.__construct_inputids(prompts, self.is_chat_model)
                            prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                    for ids in input_token_ids]
                        self.model.set_tokenizer(self.tokenizer)
                        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
                        output = self.model.generate(prompts, sampling_params)
                        response = output[0].outputs[0].text
                        self.csv_debug.get(INPUT_TOKEN_IDS, []).append(output[0].prompt_token_ids)
                        self.csv_debug.get(OUTPUT_TOKEN_IDS, []).extend([output[0].outputs[0].token_ids])
                    else:
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "1"
                        os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.data_dir
                        infer_args = {"skip_special_tokens": True}
                        generate_text, _, _ = self.pa_runner.infer(
                            [prompts],
                            self.batch_size,
                            max_new_tokens,
                            False,
                            self.is_chat_model,
                            **infer_args)
                        os.environ[ATB_LLM_TOKEN_IDS_SAVE_ENABLE] = "0"
                        response = generate_text[0]
                        if is_result:
                            input_token_ids = torch.load(
                                os.path.join(self.data_dir, 'input_ids_0.pth'),
                                weights_only=True
                            )
                            self.csv_debug.get(INPUT_TOKEN_IDS, []).append(input_token_ids.tolist())
                            with safe_open(os.path.join(self.data_dir, "output_ids_0.txt"), 'r') as f:
                                output_token_ids = list(map(int, f.read().split()))
                            self.csv_debug.get(OUTPUT_TOKEN_IDS, []).append(output_token_ids)
                    if is_result:
                        self.csv_debug[KEY].append(f"{task_name}_{i}")
                        self.csv_debug[TEST_RESULT].append(response)
                        self.csv_debug[GOLDEN_RESULT].append(data[answers])
                        self.csv_debug[LENGTH].append(data[LENGTH])
                        self.csv_debug[ALL_CLASSES].append(data[ALL_CLASSES])
                        task_result.append({"pred": response, answers: data[answers], 
                                            ALL_CLASSES: data[ALL_CLASSES], LENGTH: data[LENGTH]})
                        self.logger.debug(\
                            json.dumps({"pred": response, answers: data[answers], 
                                        LENGTH: data[LENGTH]}, ensure_ascii=False))
                except Exception as e:
                    self.error_catched += 1
                    self.logger.error("Error occurred in batch %d: %s", i, str(e), exc_info=True)
            
            if is_result:
                task_score = get_scores_by_task_name(task_name, task_result, use_longbench_e)
                task_scores.update(task_score)
                task_nums.update({f"{task_name}": len(dataset)})
                self.logger.info("dataset %d finish, result: %s, current all results: %s", 
                                idx + 1, task_score, task_scores)

        if is_result:
            final, result_total = get_final_scores(task_scores, task_nums, use_longbench_e)
            self.error_rate = self.error_catched / len(task_list)
            self.logger.info("Failed cases: %d", self.error_catched)
            self.logger.info("Failed rates: %.4f", self.error_rate)
            self.logger.debug("total result: %s", final)
            self.__save_debug()
            self.__save_result(result_total)

    def __compare_results_helper(self, result_type):
        error_1e4 = 0
        error_1e3 = 0
        total_tokens_checked = 0
        total_logits_checked = 0
        greatest_kll = 0
        for epoch_id in range(math.ceil(question_num.get(self.dataset_name, 0) / self.batch_size)):
            cnt = 0
            while True:
                golden_path = (
                    f"{self.data_dir}/GPU/{self.dataset_name}/batch{self.batch_size}/"
                    f"{epoch_id}/{result_type}_{cnt}.pth"
                )
                npu_path = (
                    f"{self.data_dir}/NPU/{self.dataset_name}/batch{self.batch_size}/"
                    f"{epoch_id}/{result_type}_{cnt}.pth"
                )
                golden_file_exists = os.path.exists(golden_path)
                npu_file_exists = os.path.exists(npu_path)
                if not golden_file_exists and not npu_file_exists:
                    self.logger.debug("%s batch %d epoch %d %s compare finish, total %d %s",
                                      self.dataset_name, self.batch_size, epoch_id, result_type, cnt, result_type)
                    break
                elif golden_file_exists and npu_file_exists:
                    golden_results = torch.load(golden_path, weights_only=True).cpu()
                    npu_results = torch.load(npu_path, weights_only=True).cpu()
                    if result_type == "tokens":
                        for i, golden_result in enumerate(golden_results):
                            total_tokens_checked += 1
                            if self.eos_token[self.batch_size * epoch_id + i] == -1 and (
                                    npu_results[i] != golden_result or npu_results[
                                i] == self.tokenizer.eos_token_id):
                                self.eos_token[self.batch_size * epoch_id + i] = cnt
                                self.logger.debug(
                                    "%s batch %d epoch %d question %d token No. %d "
                                    "is the first different token or eos token,"
                                    "ignore checking the rest.\ngolden tokenId: %s, npu tokenId: %s",
                                    self.dataset_name, self.batch_size, epoch_id, self.batch_size * epoch_id + i, 
                                    cnt, golden_result, npu_results[i])

                    elif result_type == "logits":
                        split_golden_results = torch.split(golden_results, 1, dim=0)
                        split_npu_results = torch.split(npu_results, 1, dim=0)
                        for i, split_golden_result in enumerate(split_golden_results):
                            eos_token = self.eos_token[self.batch_size * epoch_id + i]
                            if eos_token != -1 and cnt > eos_token:
                                continue
                            total_logits_checked += 1
                            golden_results_logsoftmax = torch.log_softmax(split_golden_result.float(), dim=-1)
                            npu_results_logsoftmax = torch.log_softmax(split_npu_results[i].float(), dim=-1)

                            kl_loss = torch.nn.KLDivLoss(log_target=True, reduction='sum')
                            output = kl_loss(npu_results_logsoftmax, golden_results_logsoftmax)
                            greatest_kll = output.item() if output.item() > greatest_kll else greatest_kll
                            if (output > 0.0001):
                                if (output > 0.001):
                                    error_1e3 += 1
                                error_1e4 += 1
                                self.logger.debug(
                                    "--------------------------------%s Error Begins"
                                    "--------------------------------", result_type)
                                self.logger.debug(
                                     + "%s batch %d epoch %d question %d logits No. %d fail, KL loss is: %.6f",
                                     self.dataset_name, self.batch_size, epoch_id, 
                                     self.batch_size * epoch_id + i, cnt, output.item())

                                golden_logits_sorted = torch.sort(split_golden_result, descending=True)
                                npu_logits_sorted = torch.sort(split_npu_results[i], descending=True)
                                self.logger.debug(
                                    "golden logits: \n%s\nnpu logits: \n%s", 
                                    golden_logits_sorted[0], npu_logits_sorted[0])
                                self.logger.debug(
                                    "golden index: \n%s\nnpu index: \n%s", 
                                    golden_logits_sorted[1], npu_logits_sorted[1])
                                self.logger.debug(
                                    "--------------------------------%s Error Ends"
                                    "--------------------------------", result_type)
                    cnt += 1
                else:
                    self.logger.debug("%s batch %d epoch %d %s size not equal", 
                                      self.dataset_name, self.batch_size, epoch_id, result_type)
                    self.logger.debug("%s batch %d epoch %d %s compare finish, total %d %s", 
                                      self.dataset_name, self.batch_size, epoch_id, result_type, cnt, result_type)
                    break

        if result_type == "tokens":
            self.logger.debug("%s batch %d finished check, total tokens num %d, "
                              "find %d question responses have %s mismatch",
                              self.dataset_name, self.batch_size, total_tokens_checked,
                              len(self.eos_token) - self.eos_token.count(-1), result_type)
        elif result_type == "logits":
            pass_rate = error_1e4 / total_logits_checked
            pass_result = PASS
            if pass_rate > 0.005 or error_1e3 > 0:
                pass_result = "Fail"
            self.logger.debug("%s batch %d finished check, total logits checked %d, "
                              "%d 1e-4 %s errors found, %d 1e-3 %s errors found, 1e-4 error rate %f",
                              self.dataset_name, self.batch_size, total_logits_checked, 
                              error_1e4, result_type, error_1e3, result_type, pass_rate)
            csv_result = [
                str(self.model_name).ljust(15), 
                str(self.dataset_name).ljust(15), 
                str(self.batch_size).ljust(15), 
                str(total_logits_checked).ljust(15), 
                str(round(greatest_kll, 10)).ljust(15), 
                str(round(pass_rate, 10)).ljust(15), 
                str(pass_result).ljust(15)
            ]
            csv_simplified_path = os.path.join(self.script_path, "../result", "simplified_test_result.csv")
            if not os.path.exists(csv_simplified_path):
                self.logger.warning("simplified dataset result csv file not exist, skip recording results")
                raise RuntimeError("csv result file not exist")
            with safe_open(csv_simplified_path, 'a', newline='') as csv_simplified_file:
                csv_writer = csv.writer(csv_simplified_file, delimiter='|')
                csv_writer.writerow(csv_result)
                self.logger.info("%s %s batch %d result saved in result/simplified_test_result.csv", 
                                 self.model_name, self.dataset_name, self.batch_size)

    def __compare_full_dataset_results(self):
        golden_name = '_'.join([self.model_name, self.dataset_name])
        golden_path = ''
        for file_name in os.listdir(f"{self.data_dir}/GPU/{self.dataset_name}/batch{self.batch_size}"):
            if file_name.startswith(f"{golden_name}"):
                golden_path = os.path.join(f"{self.data_dir}/GPU/{self.dataset_name}/batch{self.batch_size}", file_name)
                break

        if not os.path.exists(f"{self.current_result_path}"):
            raise RuntimeError(
                "NPU test data not exist, An error occurred in the test")
        if not os.path.exists(f"{golden_path}"):
            raise RuntimeError(
                "GPU golden data not exist, upload to result dir folder")
        result_df = pd.read_csv(self.current_result_path, sep='|', skipinitialspace=True).rename(
            columns=lambda x: x.strip())
        result_df = result_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        golden_df = pd.read_csv(golden_path, sep='|', skipinitialspace=True).rename(columns=lambda x: x.strip())
        golden_df = golden_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        csv_result = []
        if self.dataset_name == 'MMLU' or self.dataset_name == 'CEval' or self.dataset_name == 'GSM8K':
            result_total = result_df.loc[result_df['file_name'] == 'total', 'value'].values[0]
            golden_total = golden_df.loc[golden_df['file_name'] == 'total', 'value'].values[0]
            diff_val = golden_total - result_total
            pass_result = PASS
            if diff_val <= 0.1:
                self.logger.debug(
                    "%s is pass(%s%%), golden: %s, test: %s", 
                    self.current_result_path, diff_val, golden_total, result_total)
            else:
                pass_result = "Fail"
                self.logger.debug(
                    "%s is failed(%s%%), golden: %s, test: %s", 
                    self.current_result_path, diff_val, golden_total, result_total)
            csv_result = [
                str(self.model_name).ljust(15), 
                str(self.dataset_name).ljust(15), 
                str(self.batch_size).ljust(15), 
                str(round(golden_total, 10)).ljust(15), 
                str(round(result_total, 10)).ljust(15), 
                str(pass_result).ljust(15)
            ]
        elif self.dataset_name == 'TruthfulQA':
            if len(result_df) != len(golden_df):
                raise RuntimeError(f"result_df len:{len(result_df)}, golden_df len:{len(golden_df)}")
            result_mc1_sum = 0
            result_mc2_sum = 0
            golden_mc1_sum = 0
            golden_mc2_sum = 0
            pass_result = PASS
            for index, result_row in result_df.iterrows():
                golden_row = golden_df.iloc[index]
                result_mc1_sum += result_row['MC1']
                result_mc2_sum += result_row['MC2']
                golden_mc1_sum += golden_row['MC1']
                golden_mc2_sum += golden_row['MC2']
            diff_mc1 = (golden_mc1_sum - result_mc1_sum) / len(result_df)
            diff_mc2 = (golden_mc2_sum - result_mc2_sum) / len(result_df)
            if ((diff_mc1 <= 0.1) and (diff_mc2 <= 0.1)):
                self.logger.debug(
                    "%s is pass(MC1: %s MC2: %s), golden: %f , test: %f",
                    self.current_result_path, diff_mc1, diff_mc2,
                    golden_mc2_sum / len(result_df), result_mc2_sum / len(result_df))
            else:
                pass_result = "Fail"
                self.logger.debug(
                    "%s is failed(MC1: %s MC2: %s), golden: %f, test: %f", 
                    self.current_result_path, diff_mc1, diff_mc2, 
                    golden_mc2_sum / len(result_df), result_mc2_sum / len(result_df))
            csv_result = [
                str(self.model_name).ljust(15), 
                str(self.dataset_name).ljust(15), 
                str(self.batch_size).ljust(15), 
                str(round((golden_mc2_sum / len(result_df)), 10)).ljust(15), 
                str(round((result_mc2_sum / len(result_df)), 10)).ljust(15), 
                str(pass_result).ljust(15)
            ]
        csv_full_path = os.path.join(self.script_path, "../result", "full_test_result.csv")
        if not os.path.exists(csv_full_path):
            self.logger.warning("full dataset result csv file not exist, skip recording results")
            raise RuntimeError("csv result file not exist")
        with safe_open(csv_full_path, 'a', newline='') as csv_full_file:
            csv_writer = csv.writer(csv_full_file, delimiter='|')
            csv_writer.writerow(csv_result)
            self.logger.info("%s %s batch %d result saved in result/full_test_result.csv", 
                             self.model_name, self.dataset_name, self.batch_size)

    def __get_model_or_runner(self, input_length, output_length, warmup_bs=0):
        auto = "auto"
        if self.hardware_type == "NPU":
            if self.model_type == "fa":
                input_dict = {
                    'rank':self.rank,
                    'world_size':self.world_size,
                    'local_rank':self.local_rank,
                    'model_path': self.weight_dir,
                    'batch_size':self.batch_size,
                    'max_position_embeddings': (self.max_position_embedding 
                                            if self.max_position_embedding != -1 
                                            else input_length + output_length),
                    'max_input_length': input_length,
                    'max_output_length': output_length,
                    'trust_remote_code': self.trust_remote_code                   
                }
                self.fa_runner = FARunner(**input_dict)
                model_dtype = self.fa_runner.model.dtype
            else:
                max_position_embeddings = 'max_position_embeddings'
                prefill_batch_size = self.batch_size if self.prefill_batch_size == 0 else self.prefill_batch_size
                input_dict = {
                    'rank': self.rank,
                    'local_rank': self.local_rank,
                    'world_size': self.world_size,
                    'max_prefill_tokens': -1,
                    'block_size': self.block_size,
                    'model_path': self.weight_dir,
                    max_position_embeddings: (self.max_position_embedding 
                                              if self.max_position_embedding != -1 
                                              else input_length + output_length),
                    'max_prefill_batch_size': prefill_batch_size,
                    'max_batch_size': warmup_bs if warmup_bs != 0 else self.batch_size,
                    'max_input_length': input_length,
                    'max_output_length': output_length,
                    'kw_args': self.kw_args,
                    'trust_remote_code': self.trust_remote_code
                }
                if self.model_name == "qwen2_72b" or self.model_name == "qwen2_7b":
                    input_dict[max_position_embeddings] = None
                self.pa_runner = PARunner(**input_dict)
                model_dtype = self.pa_runner.model.dtype
                self.tokenizer = self.pa_runner.model.tokenizer
            user_dtype = dtype_map.get(self.data_type, None)
            if user_dtype != model_dtype:
                self.logger.error(
                    "Inconsistent dtype: Input dtype: %s, model weight dtype: %s. please check", 
                    user_dtype, model_dtype)
                raise RuntimeError(
                    f"Inconsistent dtype: Input dtype: {user_dtype}, " +
                    f"model weight dtype: {model_dtype}. please check")
            if self.model_type == "fa":
                self.logger.info('%d fa_runner: %s', self.rank, self.fa_runner)
            else:
                self.logger.info('%d pa_runner: %s', self.rank, self.pa_runner)
        else:
            self.tokenizer_params = {}
            self.set_fa_tokenizer_params()
            self.tokenizer = self.__get_fa_tokenizer(**self.tokenizer_params)
            if "starcoder" in self.model_name or "mixtral" in self.model_name:
                self.tokenizer.pad_token = "[PAD]"
            elif "llama" in self.model_name or "yi" in self.model_name or "vicuna" in self.model_name:
                self.tokenizer.pad_token_id = 0

            if LONGBENCH in self.dataset_name or \
                (NEEDLEBENCH in self.dataset_name and QWEN in self.model_name):
                self.model = LLM(model=self.weight_dir, tensor_parallel_size=self.world_size, 
                                 dtype=auto, enforce_eager=True)
            elif QWEN in self.model_name:
                self.model = AutoModelForCausalLM.from_pretrained(self.weight_dir, device_map=auto,
                                                                  torch_dtype=auto,
                                                                  trust_remote_code=self.trust_remote_code)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.weight_dir, device_map=auto,
                                                                  torch_dtype=auto, 
                                                                  trust_remote_code=self.trust_remote_code)
            if LONGBENCH not in self.dataset_name and \
                (NEEDLEBENCH not in self.dataset_name or QWEN not in self.model_name):
                self.model.generation_config = self.__remove_part_of_generation_config(self.model.generation_config)
            
            if "baichuan" in self.model_name and self.model.config.vocab_size == 64000:
                self.tokenizer.pad_token_id = 0

            if LONGBENCH in self.dataset_name or (NEEDLEBENCH in self.dataset_name and QWEN in self.model_name):
                self.logger.info("current dtype: %s", self.model.llm_engine.model_config.dtype)
            else:
                self.logger.info("current dtype: %s", self.model.dtype)

    def __construct_inputids(self, inputs, is_chat_model=False, is_truncation=False):
        def build_inputs(input_builder, conversations, **kwargs):
            return [input_builder.make_context(self.rank, conversation, **kwargs) for conversation in conversations]
        input_texts, input_ids, input_conversations = [], [], []
        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], str):
                input_texts = inputs
            elif isinstance(inputs[0], torch.Tensor):
                input_ids = inputs
            elif isinstance(inputs[0], list) and inputs[0]:
                if isinstance(inputs[0][0], int):
                    input_ids = inputs
                elif isinstance(inputs[0][0], dict):
                    input_conversations = inputs
        if not inputs:
            raise ValueError(f"The conversations must be as List[str], List[torch.Tensor], List[List[int]]"
                            f" or List[List[Dict]]. Now the inputs ({inputs}) is not acceptable or is empty")
        if is_chat_model:
            if input_conversations:
                input_ids = build_inputs(self.tokenizer_wrapper.input_builder, input_conversations)
            elif input_texts:
                input_conversations = [[{"role":"user", "content": t}] for t in input_texts]
                input_ids = build_inputs(self.tokenizer_wrapper.input_builder, input_conversations)
            else:
                print_log(self.rank, logger.warning, "Neither conversations nor input_texts exist, "
                                                     "'chat' parameter is not effective")
        elif input_texts:
            input_ids = [self.tokenizer([text], return_tensors="pt", truncation=is_truncation)["input_ids"].flatten()
                    for text in input_texts]
        return input_ids

    def __get_rank(self):
        if self.hardware_type == "GPU":
            return torch.cuda.current_device()
        else:
            if self.model_type == "fa":
                return self.fa_runner.rank
            else:
                return self.pa_runner.rank

    def __patch_hf_transformers_utils(self):
        transformers_path = transformers.__path__[0]
        transformers_utils_path = f"{transformers_path}/generation/utils.py"
        shutil.copy(transformers_utils_path, f"{transformers_path}/generation/utils_backup.py")
        with safe_open(transformers_utils_path, "r") as utils_file:
            utils_content = utils_file.readlines()
        try:
            utils_content.index(UTILS_CODE_INSERTED_MARKER)
        except ValueError:
            try:
                insert_position = utils_content.index(UTILS_CODE_MARKER)
            except ValueError as e:
                self.logger.error("UTILS_CODE_MARKER not found in the transformers utils.py file.")
                raise RuntimeError("UTILS_CODE_MARKER not found in the transformers utils.py file.") from e
            utils_content.insert(insert_position + 234, UTILS_CODE_INSERTED_PART_4)
            utils_content.insert(insert_position + 203, UTILS_CODE_INSERTED_PART_3)
            utils_content.insert(insert_position + 154, UTILS_CODE_INSERTED_PART_2)
            utils_content.insert(insert_position + 153, UTILS_CODE_INSERTED_PART_1)

            with safe_open(transformers_utils_path, "w") as utils_file:
                utils_file.writelines(utils_content)
            self.logger.info("transformers utils.py update success")
            return
        self.logger.warning("transformers utils.py not update. Please confirm it performs as you expect")

    def __setup_model_parallel(self):
        if self.hardware_type in communication_map:
            torch.distributed.init_process_group(communication_map[self.hardware_type])
        else:
            self.logger.error("unsupported hardware type")
            raise RuntimeError("unsupported hardware type")
        self.logger.info("%s distributed process init success.", communication_map.get(self.hardware_type, "Unknown"))
        if self.hardware_type == "NPU":
            self.logger.info("user npu: %d", self.rank)
            torch_npu.npu.set_device(torch.device(f"npu:{self.rank}"))
        elif self.hardware_type == "GPU":
            self.logger.info("user gpu: %d", self.rank)
            torch.cuda.set_device(self.rank)
        self.logger.info("Device Set Success!")
    
    def __get_fa_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.weight_dir, **kwargs)
        
    def __remove_part_of_generation_config(self, generation_config):
        self.logger.info("Original generation config: %s", generation_config)
        ori_gen = GenerationConfig()
        diff_dict = generation_config.to_diff_dict()
        self.logger.debug(diff_dict)
        for key in diff_dict:
            if key.endswith("_id"):
                continue
            ori_value = getattr(ori_gen, key, None)
            if ori_value is not None:
                setattr(generation_config, key, getattr(ori_gen, key))
                self.logger.info("replace %s", key)
        self.logger.info("Generation config after remove: %s", generation_config)
        return generation_config
    
    def __create_folder(self, folder_path):
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path, ignore_errors=True)
            except Exception as e:
                self.logger.error("Error deleting folder %s: %s", folder_path, e)
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(folder_path):
            self.logger.error("folder %s create fail", folder_path)
            raise RuntimeError(f"folder {folder_path} create fail")

    def __save_debug(self):        
        if self.test_type == "performance":
            debug_info_path = os.path.join(
                self.debug_dir, 
                "%s_%s_batch%d_tp%d_%s_debug_info.csv" % (
                    self.test_mode,
                    self.model_type,
                    self.batch_size,
                    self.world_size,
                    self.formatted_datetime
                )
            )
        else:
            debug_info_path = os.path.join(
                self.debug_dir, 
                "%s_%s_batch%d_tp%d_%s_%s_debug_info.csv" % (
                    self.dataset_name, 
                    self.model_type, 
                    self.batch_size, 
                    self.world_size, 
                    self.test_mode, 
                    self.formatted_datetime
                )
            )      
        df = pd.DataFrame(self.csv_debug)
        df.to_csv(debug_info_path, index=False, encoding='utf-8')
        self.logger.info("%s debug info saved to: %s", self.dataset_name, debug_info_path)

    def __save_result(self, result):
        def align_columns(df):
            max_widths = df.applymap(lambda x: len(str(x))).max()
            for col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).ljust(max_widths[col]))
            return df

        def align_headers(df):
            max_widths = [max(len(str(col)), df[col].map(lambda x: len(str(x))).max()) for col in df.columns]
            headers = [col.ljust(max_widths[i]) for i, col in enumerate(df.columns)]
            df.columns = headers
            for i, row in enumerate(df.values):
                df.iloc[i] = [str(val).ljust(max_widths[j]) for j, val in enumerate(row)]
            return df

        if "HumanEval" in self.dataset_name:
            self.csv_path = self.csv_path.replace("HumanEval_X", self.dataset_name)
            self.csv_path = self.csv_path.replace("result.csv", "infer.jsonl")
            with safe_open(self.csv_path, 'wb') as fp:
                for x in result:
                    fp.write((json.dumps(x) + "\n").encode('utf-8'))
        elif self.dataset_name == "CMMLU":
            cmmlu_eval.get_results(self.debug_dir, self.csv_path)
        elif self.dataset_name == "TruthfulQA":
            result.to_csv(self.csv_path)
        else:
            df = pd.DataFrame(result, columns=['file_name', 'value', 'correct', 'sum'])
            df = align_columns(df)
            df = align_headers(df)
            df.to_csv(self.csv_path, index=False)
        self.logger.info("%s result saved to: %s", self.dataset_name, self.csv_path)

    def __prepare_inputs_with_lora(self, idx, queries):
        inputs_dict = []
        for i, query in enumerate(queries):
            adapter = self.lora_adapter.get(str(idx + i))
            inputs_dict.append({"prompt": query, "adapter": adapter})
        return inputs_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Model test arguments")
    parser.add_argument(
        "--model_type",
        type=str,
        default='pa',
        choices=['fa', 'pa'],
        help="Specify which model type to test"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default='fp16',
        choices=['fp16', 'bf16'],
        help="Specify which datat type to test"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default='performance',
        choices=['simplified', 'full', 'precision_single', 'performance', 'performance_maxbs', "performance_single"],
        help="Specify the mode in which to run the test"
    )
    parser.add_argument("--model_name", type=str, required=True, help="name of model")
    parser.add_argument("--weight_dir", type=str, required=True, help="path to model weight folder")
    parser.add_argument("--output_dir", type=str, help="path to save the output")
    parser.add_argument("--dataset_name", type=str, default="GSM8K", help="which dataset to run")
    parser.add_argument("--shot", type=int, help="speicify dataset test few shots")
    parser.add_argument("--prefill_batch_size", type=int, default=0, help="prefill batch size")
    parser.add_argument("--batch_size", type=str, default="1", help="batch size")
    parser.add_argument("--device_id", type=int, default=7, help="device id")
    parser.add_argument("--hardware_type", type=str, default="NPU", help="current device type, GPU or NPU")
    parser.add_argument("--case_pair", type=str, default="[[256, 256], [512, 512], [1024, 1024], [2048, 2048]]",
                        help="performance test pair")
    parser.add_argument("--time_limit", type=int, help="time limit when testing performance max batch_size")
    parser.add_argument("--max_position_embeddings", type=int, help="specify whether llama model use refactor")
    parser.add_argument("--input_text_or_file", type=str, 
                        help="input_text_or_file used to test performance or precision")
    parser.add_argument("--is_chat_instruct_model", type=str, default="base",
                        help="specify whether the model use chat or instruct version")
    parser.add_argument("--context_length", type=str, default="4k", help="specify context length for dataset test")
    parser.add_argument("--lora_data_path", type=str, default="", help="a file contains a list of lora adatper name")
    parser.add_argument('--kw_args', type=str, default='', help='json input')
    parser.add_argument("--trust_remote_code", type=int, default=0, help="whether setting trust_remote_code")

    return parser.parse_args()


def get_args():
    args = parse_args()

    output_dir = (os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs") 
                  if args.output_dir is None 
                  else args.output_dir)
    case_pair = "[[256, 256], [512, 512], [1024, 1024], [2048, 2048]]" if args.case_pair == "[]" else args.case_pair
    is_chat_model = True if args.is_chat_instruct_model in ["chat", "instruct"] else False
    is_instruct_model = True if args.is_chat_instruct_model == "instruct" else False
    trust_remote_code = True if args.trust_remote_code == 1 else False
    lora_adapter = {}
    if args.lora_data_path:
        lora_data_path = standardize_path(args.lora_data_path)
        check_file_safety(lora_data_path)
        with safe_open(lora_data_path, mode="r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                lora_adapter.update(data)

    return [args.model_type, args.data_type, args.test_mode, args.model_name, output_dir, args.dataset_name,
            args.batch_size, args.device_id, args.hardware_type, case_pair, args.weight_dir,
            args.time_limit, args.max_position_embeddings, args.input_text_or_file,
            is_chat_model, is_instruct_model, args.shot, lora_adapter, args.context_length,
            args.kw_args, args.prefill_batch_size, trust_remote_code]
