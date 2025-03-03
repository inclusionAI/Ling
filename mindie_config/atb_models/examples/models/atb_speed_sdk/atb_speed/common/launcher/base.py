#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
common launcher
"""
import inspect
import logging
import os
import time
from abc import abstractmethod
from enum import Enum
from typing import Dict

import torch
from atb_speed.common.config import atb_speed_config
from atb_speed.common.log.logging import init_logger
from transformers import GenerationConfig


class DeviceType(str, Enum):
    npu = "npu"
    cuda = "cuda"
    cpu = "cpu"


def get_device() -> str:
    """
    获取当前所在设备
    :return:
    """
    flag = torch.cuda.is_available()
    if flag:
        return DeviceType.cuda
    try:
        import torch_npu
        flag = torch_npu.npu.is_available()
    except ImportError:
        flag = False
    return DeviceType.npu if flag else DeviceType.cpu


class BaseLauncher:
    """
    BaseLauncher
    """

    def __init__(self, device_ids: str = None, model_path="", options=None, trust_remote_code:bool = False):
        options = {} if options is None else options
        self.model_path = atb_speed_config.model.model_path if not model_path else model_path
        self.init_local_rank, self.init_world_size = 0, 1
        if device_ids is None and atb_speed_config.model:
            device_ids = atb_speed_config.model.device_ids
        self.device_ids = device_ids
        self.device_id_list = [int(item.strip()) for item in self.device_ids.split(",") if item.isnumeric()]
        self.local_rank, self.world_size = self.setup_model_parallel()
        self.trust_remote_code = trust_remote_code
        self.logger_name = f"device{self.local_rank}_{self.world_size}_{time.time()}.log"
        os.makedirs(atb_speed_config.model.log_dir, exist_ok=True)
        self.logger_path = os.path.join(atb_speed_config.model.log_dir, self.logger_name)
        self.logger = init_logger(logging.getLogger(f"device_{self.local_rank}"), self.logger_path)
        if atb_speed_config.model.bind_cpu:
            try:
                self.bind_cpu()
            except Exception as err:
                self.logger.error("Failed to bind cpu, skip to bind cpu. \nDetail: %s ", err)
        self.set_torch_env(self.device_ids, options)
        self.model, self.tokenizer = self.init_model()
        self.logger.info(self.model.device)
        self.logger.info("load model from %s successfully!", os.path.basename(inspect.getmodule(self.model).__file__))
        self.logger.info("load model from %s successfully!", os.path.realpath(inspect.getmodule(self.model).__file__))

    @property
    def _device(self) -> str:
        """
         获取当前所在设备
        :return:
        """
        return get_device()

    @property
    def device(self) -> torch.device:
        """
        获取模型所在的设备
        :return:
        """
        return self.model.device

    @property
    def device_type(self) -> str:
        """
        获取模型所在的设备的字符串
        :return:
        """
        return self.model.device.type

    @property
    def device_name(self) -> str:
        """
        获取所在设备的详细硬件名称
        :return:
        """
        if self.device_type == DeviceType.npu:
            device_name = torch.npu.get_device_name()
        elif self.device_type == DeviceType.cuda:
            device_name = torch.cuda.get_device_name()
        else:
            device_name = "cpu"
        return "_".join(device_name.split())

    @staticmethod
    def set_torch_env(device_ids, options: Dict = None):
        """

        :param device_ids:
        :param options:
        :return:
        """
    
    @abstractmethod
    def init_model(self):
        """
        模型初始化
        :return:
        """
        ...

    def infer(self, query, model_params=None):
        """
        推理代码
        :param query:
        :param model_params:
        :return:
        """
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            start_time = time.time()
            model_params = model_params if model_params is not None else {}
            pred = self.model.generate(**inputs, **model_params)
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        self.logger.info(output)
        self.logger.info("cost %s s", time_cost)
        new_tokens = len(pred[0]) - len(inputs.input_ids[0])
        final_msg = f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s)"
        self.logger.info(final_msg)
        return output

    def infer_batch(self, query, model_params=None):
        """
        推理代码
        :param query:
        :param model_params:
        :return:
        """
        inputs = self.tokenizer(query, return_tensors='pt', padding=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            start_time = time.time()
            model_params = model_params if model_params is not None else {}
            pred = self.model.generate(**inputs, **model_params)
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for ind, item in enumerate(output):
            self.logger.info("###### batch %s ", ind)
            self.logger.info(item)

        self.logger.info("cost %s s", time_cost)
        new_tokens = len(pred[0]) - len(inputs.input_ids[0])
        final_msg = f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s)"
        self.logger.info(final_msg)
        return output

    def infer_test(self, batch_size: int = 1, seq_in: int = 2048, seq_out: int = 64):
        """
        推理代码
        :param batch_size: 特定batch size
        :param seq_in:  特定长度输入
        :param seq_out: 特定长度输出
        :return:
        """
        inputs = self.tokenizer("hi", return_tensors='pt')
        dummy_input_ids_nxt = torch.randint(0, self.model.config.vocab_size, [batch_size, seq_in], dtype=torch.int64)
        dummy_attention_mask = torch.ones((batch_size, seq_in), dtype=torch.int64)
        inputs["input_ids"] = dummy_input_ids_nxt
        inputs["attention_mask"] = dummy_attention_mask
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            start_time = time.time()
            pred = self.model.generate(**inputs, max_new_tokens=seq_out,
                                       eos_token_id=self.model.config.vocab_size * 2)
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        self.logger.info("cost %s s", time_cost)
        new_tokens = len(pred[0]) - seq_in
        final_msg = (f"generate {batch_size * new_tokens} new tokens，"
                     f"({batch_size * new_tokens / time_cost:.2f} tokens/s)")
        self.logger.info(final_msg)
        return output

    def remove_part_of_generation_config(self, generation_config) -> GenerationConfig:
        """
        移除部分当前不支持后处理相关参数
        :param generation_config:
        :return:
        """
        ori_gen = GenerationConfig()
        diff_dict = generation_config.to_diff_dict()
        self.logger.info(diff_dict)
        for key in diff_dict:
            if key.endswith("_id"):
                continue
            ori_value = getattr(ori_gen, key, None)
            if ori_value is not None:
                setattr(generation_config, key, getattr(ori_gen, key))
                self.logger.info("replace %s", key)
        return generation_config

    def setup_model_parallel(self):
        return self.init_local_rank, self.init_world_size
    
    def pass_bind_cpu(self):
        pass
    
    def bind_cpu(self):
        self.pass_bind_cpu()