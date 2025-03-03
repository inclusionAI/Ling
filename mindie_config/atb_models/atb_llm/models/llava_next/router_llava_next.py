# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np
from transformers import AutoProcessor

from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from ..base.router import BaseRouter
from .flash_causal_llava_next import LlavaNextConfig
from ..llama.config_llama import LlamaConfig
from ..base.config import QuantizationConfig
from ..base.model_utils import safe_get_tokenizer_from_pretrained, safe_from_pretrained
from .data_preprocess_llava_next import data_prerocess_llava_next
from .input_builder_llava_next import LlavaNextInputBuilder, get_token_ids


_VIDEO_TOKEN_ID = -2


def from_list_format(list_format: List[Dict], image_token_str: str, video_token_str: str):
    result_promt = "User: "
    text = ""
    image_or_video = ""
    for ele in list_format:
        if "image" in ele:
            if image_token_str is None:
                raise ValueError(f"The model seems dosen't support image inputs.")
            image_or_video += image_token_str + "\n"
        elif "video" in ele:
            if video_token_str is None:
                raise ValueError(f"The model seems dosen't support video inputs.")
            image_or_video += video_token_str + "\n"
        elif "text" in ele:
            text += ele["text"]
        else:
            raise KeyError("Unsupport element: " + str(ele))
    if image_or_video != "":
        result_promt += image_or_video + "\n"
    result_promt += text + "ASSISTANT:"
    return result_promt


@dataclass
class Llava_nextRouter(BaseRouter):
    _processor: Any = None
    
    @property
    def model_chat_template_version(self):
        if "mistral" in self.config.text_model_name:
            if self.config.model_type == "llava_next_video":
                return "default"
            return "mistral"
        if "NousResearch" in self.config.text_model_name:
            return "v1.6-34b"
        if "Qwen" in self.config.text_model_name:
            return "qwen"
        if "llama" in self.config.text_model_name:
            return "llama"
        return "default"

    def tokenize(self, inputs, **kwargs):
        shm_name_save_path = kwargs.get("shm_name_save_path", None)
        image_or_video_path_list = [
            single_input.get("image_or_video") or single_input.get("image") or single_input.get("video")
            for single_input in inputs
            if "image_or_video" in single_input or "image" in single_input or "video" in single_input
        ]
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index if self.config.video_token_index else _VIDEO_TOKEN_ID
    
        image_token_str = self.tokenizer.decode(image_token_id)
        video_token_str = self.tokenizer.decode(video_token_id) if self.config.video_token_index else None

        query = from_list_format(inputs, image_token_str, video_token_str)
        new_input_ids = get_token_ids(query, image_or_video_path_list, self.config, 
            self.processor(), shm_name_save_path)
        return new_input_ids

    def check_config_llava(self, config):
        super().check_config(config)
        attribute_ranges = {
            'mm_hidden_size': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_config(self):
        config = LlavaNextConfig.from_pretrained(self.model_name_or_path)
        config.text_config = LlamaConfig.from_dict(config.text_config)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_llava(config)
        config.model_name_or_path = self.model_name_or_path
        return config

    def processor(self):
        if self._processor is not None:
            return self._processor
        if self.config.text_config.num_hidden_layers == 60 and \
            self.config.architectures[0] == "LlavaNextForConditionalGeneration":
            self._processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path, use_fast=False)
        else:
            self._processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path)
        return self._processor
        
    def is_llava_next34b(self):
        config = self.get_config()
        num_hidden_layers = config.text_config.num_hidden_layers
        architectures = config.architectures
        if num_hidden_layers in [60] and architectures[0] == "LlavaNextForConditionalGeneration":
            return True
        return False
    
    def get_tokenizer(self):
        use_fast = True
        if self.is_llava_next34b():
            use_fast = False 
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast,
        )

    def get_input_builder(self):
        return LlavaNextInputBuilder(self.processor(), self.model_chat_template_version, self.config)