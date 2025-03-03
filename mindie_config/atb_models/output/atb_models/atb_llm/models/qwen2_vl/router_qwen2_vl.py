# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from transformers import AutoTokenizer, AutoImageProcessor

from ..base.config import QuantizationConfig
from ..base.model_utils import safe_from_pretrained
from ..base.router import BaseRouter
from ..qwen2_vl.config_qwen2_vl import Qwen2vlConfig
from ..qwen2_vl.data_preprocess_qwen2_vl import fetch_image, fetch_video
from ..qwen2_vl.input_builder_qwen2_vl import Qwen2vlInputBuilder

VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
PYTORCH_TENSOR = "pt"
_IMAGE = "image"
_VIDEO = "video"
_TEXT = "text"

messages_template = [
    {
        "role": "user",
        "content": [],
    }
]
# 服务化支持torch分布式所需环境变量
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '5678')


def process_shared_memory(pixel_values, shm_name_save_path, grid_thw):
    shm = create_shm(pixel_values.nbytes, shm_name_save_path)
    shared_array = np.ndarray(pixel_values.shape, dtype=np.uint8, buffer=shm.buf)
    shared_array[:] = pixel_values
    shm_name = encode_shm_name_to_int64(shm.name)
    shape_value = encode_shape_to_int64(pixel_values.shape)
    thw_value = encode_shape_to_int64(grid_thw[0])
    return shm_name, shape_value, thw_value


@dataclass
class Qwen2vlRouter(BaseRouter):
    _image_processor: Any = None

    @property
    def image_processor(self):
        if not hasattr(self, "_image_processor"):
            self._image_processor = self.get_image_processor()
        elif self._image_processor is None:
            self._image_processor = self.get_image_processor()
        return self._image_processor

    def tokenize(self, inputs, **kwargs):
        image_token_id = getattr(self.config, "image_token_id", _IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", _VIDEO_TOKEN_ID)

        vision_info_list = []
        message_list = []
        shm_name_save_path = kwargs.get('shm_name_save_path', None)

        for single_input in inputs:
            if single_input.get(_IMAGE, None):
                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[_IMAGE]))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")

                images_inputs, feature_lens = fetch_image(self.image_processor, single_input)

                shm_name, shape_value, thw_value = process_shared_memory(
                    images_inputs.pixel_values,
                    shm_name_save_path,
                    images_inputs.image_grid_thw
                )
                vision_info_list.append([shm_name, shape_value, thw_value, feature_lens, image_token_id])
                message_list.append(single_input)

            elif single_input.get(_VIDEO, None):
                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[_VIDEO]))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
                # 默认fps为0.5，可自行调整
                video_single_message = {"type": "video", "video": single_input[_VIDEO], "fps": 0.5}
                video_inputs, feature_lens = fetch_video(self.image_processor, video_single_message)

                shm_name, shape_value, thw_value = process_shared_memory(
                    video_inputs.pixel_values_videos,
                    shm_name_save_path,
                    video_inputs.video_grid_thw
                )
                vision_info_list.append([shm_name, shape_value, thw_value, feature_lens, video_token_id])
                message_list.append(single_input)

            elif single_input.get(_TEXT, None):
                message_list.append(single_input)
            else:
                raise TypeError("The input field currently only needs to support 'image', 'video' and 'text'.")
        return self.process_token(vision_info_list, message_list)

    def get_input_builder(self):
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            return Qwen2vlInputBuilder(self.tokenizer, self.image_processor, self.config,
                                       max_length=self.config.max_position_embeddings)
        return Qwen2vlInputBuilder(self.tokenizer, self.config)

    def get_config(self):
        config = safe_from_pretrained(Qwen2vlConfig, self.model_name_or_path)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        config.model_name_or_path = self.model_name_or_path
        self.check_config_qwen2_vl(config)
        return config

    def get_tokenizer(self):
        return safe_from_pretrained(AutoTokenizer, self.model_name_or_path)

    def get_image_processor(self):
        return safe_from_pretrained(AutoImageProcessor, self.model_name_or_path)

    def check_config_qwen2_vl(self, config):
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

    def process_token(self, vision_info_list, message_list):
        messages = copy.deepcopy(messages_template)
        messages[0]["content"] = message_list
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not prompt:
            prompt = self.tokenizer.apply_chat_template(
                message_list, tokenize=False, add_generation_prompt=True
            )
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten()
        new_input_ids = input_ids
        vision_start_token_id = getattr(self.config, "vision_start_token_id", VISION_START_TOKEN_ID)
        vision_end_token_id = getattr(self.config, "vision_end_token_id", VISION_END_TOKEN_ID)

        bos_pos = torch.where(torch.eq(input_ids, vision_start_token_id))[0]
        eos_pos = torch.where(torch.eq(input_ids, vision_end_token_id))[0]

        image_num = bos_pos.size(0)
        expand_token_ids = []
        pre = 0
        for i in range(0, image_num, 1):
            feature_lens = vision_info_list[i][3]
            text_token = input_ids[pre: bos_pos[i]]
            pre = eos_pos[i] + 1
            image_pad_token = torch.cat(
                [
                    torch.tensor([vision_start_token_id], dtype=input_ids.dtype),
                    torch.tensor([vision_info_list[i][0]], dtype=input_ids.dtype),
                    torch.tensor([vision_info_list[i][1]], dtype=input_ids.dtype),
                    torch.tensor([vision_info_list[i][2]], dtype=input_ids.dtype),
                    torch.full((feature_lens - 3,), vision_info_list[i][-1], dtype=input_ids.dtype),
                    torch.tensor([vision_end_token_id], dtype=input_ids.dtype),
                ]
            )
            if text_token.size(0) != 0:
                expand_token_ids.append(text_token)

            if image_pad_token.size(0) != 0:
                expand_token_ids.append(image_pad_token)

        text_token = input_ids[pre:]
        if text_token.size(0) != 0:
            expand_token_ids.append(text_token)

        if expand_token_ids:
            new_input_ids = torch.cat(expand_token_ids)
        return new_input_ids
