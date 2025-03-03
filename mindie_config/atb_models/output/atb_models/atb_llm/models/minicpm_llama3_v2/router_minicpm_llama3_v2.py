# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import json
from typing import Dict, List
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from atb_llm.models.minicpm_llama3_v2.flash_causal_minicpm_llama3_v2 import process_qs
from transformers import AutoProcessor
from ..base.config import QuantizationConfig
from ..base.router import BaseRouter
from .configuration_minicpm import MiniCPMVConfig
from ...utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from ..base.model_utils import safe_from_pretrained
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from ...utils.multimodal_utils import safe_open_image


EOT_ID = '<|eot_id|>'
MSG_CONTENT = 'content'

EOS_TOKEN_ID = 128009


def from_list_format(list_format: List[Dict]):
    texts = []
    img = ""
    for ele in list_format:
        if "image_url" in ele:
            img = ele["image_url"]
        if "image" in ele:
            img = ele["image"]
        if "text" in ele:
            text = ele["text"]
            texts = [{'role': 'user', 'content': text}]

    return texts, img


@dataclass
class Minicpmllama3v2Router(BaseRouter):

    def __post_init__(self):
        super().__post_init__()
        self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path, 
                                              trust_remote_code=self.trust_remote_code)
        self.tokenizer.eos_token_id = EOS_TOKEN_ID
    
    @staticmethod
    def is_openai_format(inputs):
        conversation_state = False
        for item in inputs:
            if 'role' in item:
                conversation_state = True
        return conversation_state

    @staticmethod
    def process_shm(image_pixel, shm_name_save_path, dtype=np.float32):
        shm = create_shm(image_pixel.nbytes, shm_name_save_path)
        shared_array = np.ndarray(image_pixel.shape, dtype=dtype, buffer=shm.buf)
        shared_array[:] = image_pixel
        shm_name = encode_shm_name_to_int64(shm.name)
        shape_value = encode_shape_to_int64(image_pixel.shape)
        return shm_name, shape_value
    
    def process_img_text_data(self, content):
        text, image_path = from_list_format(content)
        image = safe_open_image(Image, image_path)
        copy_text, images = process_qs(text, image)
        prompt = self.processor.tokenizer.apply_chat_template(copy_text, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, images, return_tensors="pt")
        image.close()
        return prompt, inputs, image_path
    
    def apply_openai_template(self, inputs):
        content = inputs[0]["content"]
        return self.process_img_text_data(content)

    def apply_general_template(self, inputs):
        for idx, item in enumerate(inputs):
            if "image" in item:
                image = item["image"]
                inputs[idx] = {"type":"image_url", "image_url":image}
            item_name = "text_test".split("_")[0]
            if item_name in item:
                text = item[item_name]
                inputs[idx] = {"type":item_name, item_name:text}
        return self.process_img_text_data(inputs)
    
    
    def tokenize(self, inputs, **kwargs):
        if self.is_openai_format(inputs):
            _, inputs, image_path = self.apply_openai_template(inputs)
        else:
            _, inputs, image_path = self.apply_general_template(inputs)

        image_bound = inputs['image_bound']
        tgt_sizes = inputs['tgt_sizes']
        input_ids = inputs['input_ids'][0]
        input_ids = torch.tensor(input_ids, dtype=torch.int64)

        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        shm_name_list = []
        shape_value_list = []
        image_type = "pixel_values"
        inputs = [inputs]
        for single_input in inputs:
            if image_type not in single_input.keys():
                continue
            image_pixels = single_input[image_type]
            for _, image_pixel in enumerate(image_pixels[0]):
                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
                shm_name, shape_value = self.process_shm(image_pixel, shm_name_save_path)
                shm_name_list.append(shm_name)
                shape_value_list.append(shape_value)
                
        
        start_shm_point = image_bound[0]
        shm_name, shape_value = self.process_shm(tgt_sizes[0], shm_name_save_path, dtype=np.int64)
        shm_name_list = [shm_name] + shm_name_list
        shape_value_list = [shape_value] + shape_value_list
        shm_name, shape_value = self.process_shm(image_bound[0], shm_name_save_path, dtype=np.int64)
        shm_name_list = [shm_name] + shm_name_list
        shape_value_list = [shape_value] + shape_value_list

        for i, _ in enumerate(shm_name_list):
            if start_shm_point[0][0] + 2 * i + 1 >= input_ids.size(0):
                raise ValueError("input_ids length is not enough, pelase check.")
            input_ids[start_shm_point[0][0] + 2 * i] = torch.tensor(shm_name_list[i], dtype=torch.int64)
            input_ids[start_shm_point[0][0] + 2 * i + 1] = torch.tensor(shape_value_list[i], dtype=torch.int64)

        return input_ids

    def check_config_minicpmllama3v2(self, config):
        super().check_config(config)
        attribute_ranges = {'mm_hidden_size': (1, 2147483647), 'num_key_value_heads': (1, 2147483647)}
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_config(self):
        config = safe_from_pretrained(MiniCPMVConfig, self.model_name_or_path)
        config.model_name_or_path = self.model_name_or_path
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_minicpmllama3v2(config)
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )

    def get_generation_config(self):
        generation_config = super().get_generation_config()
        generation_config["eos_token_id"] = [
            generation_config["eos_token_id"],
            self.tokenizer.convert_tokens_to_ids(EOT_ID)
        ]
        return generation_config

    def make_context(self, rank: int, conversation: List[Dict[str, List[Dict]]], 
                        system: str = "You are a helpful assistant.",
                        **kwargs):
        return self.tokenize(conversation)

    def get_input_builder(self):
        return self
    
    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        return position_ids

