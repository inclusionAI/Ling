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
from typing import Dict, List
import torch
import torch_npu
import numpy as np

from atb_llm.utils import shm_utils
from ..base.input_builder import InputBuilder
from .data_preprocess_llava_next import data_prerocess_llava_next


_CONTENT = "content"
_IMAGE = "image"
_VIDEO = "video"
_TEXT = "text"
_VIDEO_TOKEN_ID = -2


class LlavaNextInputBuilder(InputBuilder):
    def __init__(self, processor, model_version, config, **kwargs):
        self.model_version = model_version
        self.config = config
        self.processor = processor
        self.chat_templates = {
            "mistral": generate_mistral_template,
            "v1.6-34b": generate_34b_template,
            "qwen": generate_qwen_template,
            "llama": generate_llama_teplate,
            "default": generate_vicuna_template
        }
        super().__init__(processor.tokenizer, system_role_name="assistant", user_role_name="user", **kwargs)

    def make_context(
        self, 
        rank: int,
        conversation: List[Dict[str, List[Dict]]], 
        system: str = "You are a helpful assistant.",
        adapt_to_max_length: bool = False,
        **kwargs):
        if not isinstance(conversation[0]["content"], list):
            raise TypeError("The conversation \"content\" should be a List[Dict].")
        
        shm_name_save_path = kwargs.get('shm_name_save_path', None)

        context_tokens = self._apply_chat_template(
            conversation,
            shm_name_save_path=shm_name_save_path,
            )
        return context_tokens

    def _apply_chat_template(
        self,
        conversation: List[Dict[str, List[Dict]]],
        shm_name_save_path: str = None,
        **kwargs):
        query = self.chat_templates.get(self.model_version)(conversation)
        media_path_list = []
        for message in conversation:
            for single_input in message.get(_CONTENT):
                if _TEXT in single_input.keys():
                    continue
                media_path = single_input.get(_IMAGE) or single_input.get(_VIDEO) \
                    if _IMAGE in single_input or _VIDEO in single_input else None
                single_type = _IMAGE if _IMAGE in single_input else _VIDEO
                if media_path is None:
                    raise ValueError(f"The input of type {single_type} shouldn't be None.")
                media_path_list.append(media_path)
        context_tokens = get_token_ids(query, media_path_list, self.config, 
            self.processor, shm_name_save_path)
        return context_tokens


def get_token_ids(query, media_path_list, config, processor, shm_name_save_path=None):
    pad_token_id = config.text_config.pad_token_id if \
            config.text_config.pad_token_id is not None else -1
    shm_name_list = []
    shape_value_list = []
    feature_lens = []
    image_size_list = []
    media_token_ids = []
    for media_path in media_path_list:
        
        pixel_value, single_lens, image_or_video_token_id, image_size = data_prerocess_llava_next(
            processor, config, media_path)
        if shm_name_save_path is None:
            shm_name_save_dir = os.path.dirname(os.path.dirname(media_path))
            shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
        shm = shm_utils.create_shm(pixel_value.nbytes, shm_name_save_path)
        shared_array = np.ndarray(pixel_value.shape, dtype=np.float16, buffer=shm.buf)
        shared_array[:] = pixel_value
        shm_name_value = shm_utils.encode_shm_name_to_int64(shm.name)
        shape_value = shm_utils.encode_shape_to_int64(pixel_value.shape)
        image_size_value = shm_utils.encode_shape_to_int64([1, image_size[0], image_size[1]]) \
            if image_size is not None else None
        shm_name_list.append(shm_name_value)
        shape_value_list.append(shape_value)
        image_size_list.append(image_size_value)
        feature_lens.append(single_lens)
        media_token_ids.append(image_or_video_token_id)

    context_ids = processor.tokenizer(query, return_tensors="pt")["input_ids"].flatten()
    image_token_id = config.image_token_index
    video_token_id = config.video_token_index if config.video_token_index else _VIDEO_TOKEN_ID
    bos_pos = torch.nonzero((context_ids == image_token_id) | (context_ids == video_token_id)).view(-1)

    context_tokens = context_ids
    image_num = bos_pos.shape[0]
    expand_token_ids = []
    pre_index = 0
    for i in range(image_num):
        text_token = context_ids[pre_index: bos_pos[i]]
        pre_index = bos_pos[i] + 1
        if image_size_list[i] is not None:
            new_media_token = torch.cat(
                [
                    torch.tensor([media_token_ids[i]], dtype=context_ids.dtype),
                    torch.tensor([shm_name_list[i], shape_value_list[i], image_size_list[i]], dtype=context_ids.dtype),
                    torch.full((feature_lens[i] - 5, ), pad_token_id, dtype=context_ids.dtype),
                    torch.tensor([media_token_ids[i]], dtype=context_ids.dtype),
                ]
            )
        else:
            new_media_token = torch.cat(
                [
                    torch.tensor([media_token_ids[i]], dtype=context_ids.dtype),
                    torch.tensor([shm_name_list[i], shape_value_list[i]], dtype=context_ids.dtype),
                    torch.full((feature_lens[i] - 4, ), pad_token_id, dtype=context_ids.dtype),
                    torch.tensor([media_token_ids[i]], dtype=context_ids.dtype),
                ]
            )
        if text_token.size(0) != 0:
            expand_token_ids.append(text_token)
        if new_media_token.size(0) != 0:
            expand_token_ids.append(new_media_token)
            
    text_token = context_ids[pre_index:]
    if text_token.size(0) != 0:
        expand_token_ids.append(text_token)
        
    if expand_token_ids:
        context_tokens = torch.cat(expand_token_ids)
    return context_tokens


def generate_mistral_template(messages):
    output = []

    for single_message in messages:
        role = single_message.get('role')
        
        if role == 'system':
            # Handle system message
            output.append("<<SYS>>\n")
            output.append(single_message.get(_CONTENT)[0]['text'])
            output.append("\n<</SYS>>\n\n")

        elif role == 'user':
            # Handle user message
            output.append("[INST] ")

            images = []
            videos = []
            texts = []
            for content in single_message.get(_CONTENT):
                if content.get(_IMAGE):
                    images.append("<image>\n")
                elif content.get(_VIDEO):
                    videos.append("<video>\n")
                elif content.get(_TEXT):
                    texts.append(content.get(_TEXT))
                else:
                    raise ValueError(f"Unknown input type: {content.keys()}.")
            output = output + images + videos + texts
            output.append(" [/INST]")

        elif role == 'assistant':
            # Handle assistant message
            output.append(" ")
            output.append(single_message.get(_CONTENT)[0].get(_TEXT))
            output.append("<\\s> ")

        else:
            raise ValueError("Only user and assistant roles are supported!")

    # Join the output list into a single string
    return ''.join(output)


def generate_vicuna_template(messages, add_generation_prompt=True):
    output = []

    for single_message in messages:
        role = single_message.get('role')
        
        # Add role prefix if not 'system'
        if role != 'system':
            output.append(f"{role.upper()}: ")

        images = []
        videos = []
        texts = []
        for content in single_message.get(_CONTENT):
            if content.get(_IMAGE):
                images.append("<image>\n")
            elif content.get(_VIDEO):
                videos.append("<video>\n")
            elif content.get(_TEXT):
                texts.append(content.get(_TEXT) + ' ')
            else:
                raise ValueError(f"Unknown input type: {content.keys()}.")
        output = output + images + videos + texts

    # Add generation prompt if specified
    if add_generation_prompt:
        output.append("ASSISTANT:")

    # Join the output list into a single string
    return ''.join(output)


def generate_34b_template(messages, add_generation_prompt=True):
    output = []

    for single_message in messages:
        role = single_message.get('role')

        # Start message block
        output.append(f"<|im_start|>{role}\n")

        # Render images first, videos next, text based on role last
        images = []
        videos = []
        texts = []
        for content in single_message.get(_CONTENT):
            if content.get(_IMAGE):
                images.append("<image>\n")
            elif content.get(_VIDEO):
                videos.append("<video>\n")
            elif content.get(_TEXT):
                texts.append(content.get(_TEXT))
            else:
                raise ValueError(f"Unknown input type: {content.keys()}.")
        output = output + images + videos + texts

        # End message block
        output.append("<|im_end|>\n")
    # Add generation prompt if specified
    if add_generation_prompt:
        output.append("<|im_start|>assistant\n")

    # Join the output list into a single string
    return ''.join(output)


def generate_qwen_template(messages, add_generation_prompt=True):
    output = []

    for single_message in messages:
        # Start the message block
        output.append(f"<|im_start|>\n{single_message.get('role')}\n")

        images = []
        texts = []
        # Render images first
        for content in single_message.get(_CONTENT):
            if content.get(_IMAGE):
                images.append("<image>\n")
            elif content.get(_TEXT):
                texts.append(f"\n{content.get(_TEXT)}")
            else:
                raise ValueError(f"Unknown input type: {content.keys()}.")
        output = output + images + texts
        # End the message block
        output.append("<|im_end|>\n")

    # Add generation prompt if specified
    if add_generation_prompt:
        output.append("<|im_start|>assistant\n")
    # Join the output list into a single string
    return ''.join(output)


def generate_llama_teplate(messages, add_generation_prompt=True):
    output = []

    for single_message in messages:
        # Start the header block
        output.append(f"<|start_header_id|>{single_message.get('role')}<|end_header_id|>\n\n")

        images = []
        texts = []
        # Render images first
        for content in single_message.get(_CONTENT):
            if content.get(_IMAGE):
                output.append("<image>")
            elif content.get(_TEXT):
                output.append(f"\n{content.get(_TEXT)}<|eot_id|>")
            else:
                raise ValueError(f"Unknown input type: {content.keys()}.")
        output = output + images + texts
    # Add generation prompt if specified
    if add_generation_prompt:
        output.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    # Join the output list into a single string
    return ''.join(output)
