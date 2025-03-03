#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
performance test base
"""
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable

import torch
import torch.distributed as dist
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher.base import BaseLauncher
from atb_speed.common.timer import Timer
from atb_llm.utils.file_utils import safe_open


class PerfMode(str, Enum):
    detail = "detail"
    normal = "normal"


@dataclass
class PerformanceTestConfig:
    """
    PerformanceTestGPUConfig
    """
    batch_size: int = 1
    max_len_exp: int = 5
    min_len_exp: int = 11
    model_name: str = "model"
    device_name: str = "cpu"
    save_file_name: str = ""
    case_pair: List[List[int]] = None

    def __post_init__(self):
        self.batch_size = atb_speed_config.performance.batch_size
        self.max_len_exp = atb_speed_config.performance.max_len_exp
        self.min_len_exp = atb_speed_config.performance.min_len_exp
        self.model_name = atb_speed_config.performance.model_name
        self.case_pair = atb_speed_config.performance.case_pair
        if not atb_speed_config.performance.save_file_name:
            self.save_file_name = f"performance_test_{self.model_name}_{self.device_name}_bs{self.batch_size}.csv"
        else:
            self.save_file_name = atb_speed_config.performance.save_file_name


class PerformanceTest:
    """
    PerformanceTestNPU
    """

    def __init__(self, launcher: BaseLauncher):
        """

        :param launcher:
        """
        self.launcher = launcher
        self.local_rank, self.world_size = launcher.local_rank, launcher.world_size
        self.config = PerformanceTestConfig(device_name=self.launcher.device_name)
        self.launcher.logger.info(self.config.__dict__)
        self.model, self.tokenizer = launcher.model, launcher.tokenizer
        self.dummy_input = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
        if atb_speed_config.performance.perf_mode == PerfMode.detail:
            self.perf = self._perf_detail_v2
        else:
            self.perf = self._perf
        self.test_case = self.generate_test_case()

    def generate_test_case(self):
        if self.config.case_pair is None:
            return [[2 ** i, 2 ** j]
                    for i in range(self.config.min_len_exp, self.config.max_len_exp + 1)
                    for j in range(self.config.min_len_exp, self.config.max_len_exp + 1)]
        return self.config.case_pair

    def warm_up(self, seq_len_in=None, seq_len_out=None):
        """

        :return:
        """
        if seq_len_in is None:
            seq_len_in = max(case[0] for case in self.test_case)
        if seq_len_out is None:
            seq_len_out = max(case[1] for case in self.test_case)
        dummy_input_ids_nxt = torch.randint(0, self.model.config.vocab_size, [self.config.batch_size, seq_len_in],
                                            dtype=torch.int64)
        dummy_attention_mask = torch.ones((self.config.batch_size, seq_len_in), dtype=torch.int64)
        inputs = self.tokenizer([self.dummy_input] * self.config.batch_size, return_tensors="pt", padding='max_length',
                                max_length=seq_len_in)
        inputs["input_ids"] = dummy_input_ids_nxt
        inputs["attention_mask"] = dummy_attention_mask
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=seq_len_out,
                eos_token_id=self.model.config.vocab_size * 2
            )
        self.launcher.logger.info("warm up finished.")

    def run_test(self):
        self.launcher.logger.info("---------------inference---------------")
        file = None
        if self.local_rank == 0:
            file = safe_open(self.config.save_file_name, "w", encoding="utf-8")
            file.write(
                "batch_size,"
                "input_seq_len(Encoding),"
                "output_seq_len(Decoding),"
                "ResponseTime(s),"
                "forward_first_token_time(ms),"
                "forward_next_token_time(ms),"
                "pre_next_token_time(ms),"
                "post_next_token_time_post(ms)\n")
        for seq_len_in, seq_len_out in self.test_case:
            time_tensor = self._run(seq_len_in, seq_len_out)
            if self.local_rank == 0:
                file.write(
                    f"{self.config.batch_size},"
                    f"{seq_len_in},"
                    f"{seq_len_out},"
                    f"{round(time_tensor[0], 2)},"
                    f"{time_tensor[1]},"
                    f"{time_tensor[2]},"
                    f"{time_tensor[3]},"
                    f"{time_tensor[4]}\n")
        if self.local_rank == 0:
            file.close()

    def _run(self, seq_len_in, seq_len_out):
        dummy_input_ids_nxt = torch.randint(0, self.model.config.vocab_size, [self.config.batch_size, seq_len_in],
                                            dtype=torch.int64)
        dummy_attention_mask = torch.ones((self.config.batch_size, seq_len_in), dtype=torch.int64)
        inputs = self.tokenizer(
            [self.dummy_input] * self.config.batch_size,
            return_tensors="pt", padding='max_length', max_length=seq_len_in)
        inputs["input_ids"] = dummy_input_ids_nxt
        inputs["attention_mask"] = dummy_attention_mask
        inputs = inputs.to(self.model.device)
        self.launcher.logger.info("---------------inputs shape---------------")
        self.launcher.logger.info(inputs.input_ids.shape)
        self.launcher.logger.info(f"seq_len_in: {seq_len_in}, seq_len_out: {seq_len_out}")
        start_time = time.time()
        forward_first_token_time, forward_next_token_time, pre_next_token_time, post_next_token_time_post = (
            self.perf(inputs, seq_len_out))
        end_time = time.time()
        # output
        # time analysis
        total_time = end_time - start_time
        time_tensor = torch.tensor(
            [total_time,
             forward_first_token_time,
             forward_next_token_time,
             pre_next_token_time,
             post_next_token_time_post], device=self.model.device)
        if self.world_size > 1:
            dist.all_reduce(time_tensor, dist.ReduceOp.MAX)
        time_tensor = time_tensor.tolist()
        return time_tensor

    def _perf_detail_v2(self, inputs, seq_len_out):
        """
        使用装饰器的方式进行计时，从而从根本上解决侵入式修改打点的方式
        :param inputs:
        :param seq_len_out:
        :return:
        """
        Timer.reset()
        Timer.sync = getattr(torch, self.launcher.device_type).synchronize
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=seq_len_out,
                                               eos_token_id=self.model.config.vocab_size * 2  # 避免提前停止
                                               )
            # decode
            if not atb_speed_config.performance.skip_decode:
                _ = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        return [Timer.timeit_res.first_token_delay, Timer.timeit_res.next_token_avg_delay, 0, 0]

    def _perf_detail(self, inputs, seq_len_out):
        with torch.no_grad():
            generate_ids, \
                forward_first_token_time, \
                forward_next_token_time, \
                pre_next_token_time, \
                post_next_token_time_post = \
                self.model.generate(**inputs, max_new_tokens=seq_len_out,
                                    eos_token_id=self.model.config.vocab_size * 2  # 避免提前停止
                                    )
            # decode
            if not atb_speed_config.performance.skip_decode:
                _ = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        return [forward_first_token_time,
                forward_next_token_time,
                pre_next_token_time,
                post_next_token_time_post]

    def _perf(self, inputs, seq_len_out):
        with torch.no_grad():
            getattr(torch, self.launcher.device_type).synchronize()
            first_token_start = time.time()
            generate_ids = self.model.generate(**inputs,
                                               min_new_tokens=1,
                                               max_new_tokens=1)
            getattr(torch, self.launcher.device_type).synchronize()
            first_token_end = time.time()
            if not atb_speed_config.performance.skip_decode:
                _ = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)

            getattr(torch, self.launcher.device_type).synchronize()
            total_start = time.time()
            generate_ids = self.model.generate(
                **inputs,
                min_new_tokens=seq_len_out,
                max_new_tokens=seq_len_out
            )
            getattr(torch, self.launcher.device_type).synchronize()
            total_end = time.time()
        if not atb_speed_config.performance.skip_decode:
            _ = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # time analysis
        forward_first_token_time = (first_token_end - first_token_start) * 1000
        time_inc_total = (total_end - total_start) * 1000
        if seq_len_out < 2:
            raise ValueError("The seq_len_out must larger than 2!")
        forward_next_token_time = (time_inc_total - forward_first_token_time) / (seq_len_out - 1)
        return [forward_first_token_time, forward_next_token_time, 0, 0]
