# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
from typing import Dict, List

import numpy as np
import torch
from atb_llm.utils.shm_utils import decode_shape_from_int64
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm

from ..base.input_builder import InputBuilder
from ..qwen2_vl.data_preprocess_qwen2_vl import fetch_image, fetch_video

_IMAGE_START_ID = 151652
_IMAGE_END_ID = 151653
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
PYTORCH_TENSOR = "pt"
_CONTENT = "content"
_IMAGE = "image"
_VIDEO = "video"


def process_shared_memory(pixel_values, shm_name_save_path, grid_thw):
    shm = create_shm(pixel_values.nbytes, shm_name_save_path)
    shared_array = np.ndarray(pixel_values.shape, dtype=np.uint8, buffer=shm.buf)
    shared_array[:] = pixel_values
    shm_name = encode_shm_name_to_int64(shm.name)
    shape_value = encode_shape_to_int64(pixel_values.shape)
    thw_value = encode_shape_to_int64(grid_thw[0])
    return shm_name, shape_value, thw_value


class Qwen2vlInputBuilder(InputBuilder):
    def __init__(self, tokenizer, image_processor, config, **kwargs):
        self.config = config
        self.image_processor = image_processor
        super().__init__(tokenizer, **kwargs)

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        if np.any(np.equal(input_ids, _IMAGE_START_ID)):
            bos_pos = np.where(np.equal(input_ids, _IMAGE_START_ID))[0]
            eos_pos = np.where(np.equal(input_ids, _IMAGE_END_ID))[0]
            vision_num = bos_pos.shape[0]
            deltas = 0
            for i in range(vision_num):
                thw_shape_value = input_ids[bos_pos[i] + 3]
                thw_shape = decode_shape_from_int64(thw_shape_value)

                vision_feature_len = eos_pos[i] - bos_pos[i] - 1
                max_hw = max(thw_shape[1:])
                if thw_shape[0] > (max_hw // 2):
                    deltas += vision_feature_len - thw_shape[0]
                else:
                    deltas += vision_feature_len - max_hw // 2
            position_ids[-1] = position_ids[-1] - deltas
        return position_ids

    def make_context(
            self,
            rank: int,
            conversation: List[Dict[str, List[Dict]]],
            **kwargs):
        if not isinstance(conversation[0]["content"], list):
            raise ValueError("The conversation \"content\" should be a List[Dict].")
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        context_tokens = self.apply_chat_template(
            conversation,
            shm_name_save_path=shm_name_save_path,
        )
        return context_tokens

    def apply_chat_template(
            self,
            conversation: List[Dict[str, List[Dict]]],
            shm_name_save_path: str = None,
            **kwargs):

        image_token_id = getattr(self.config, "image_token_id", _IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", _VIDEO_TOKEN_ID)

        vision_info_list = []
        for single_conversation in conversation:
            for single_input in single_conversation[_CONTENT]:
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
        return self.process_token(vision_info_list, conversation)

    def process_token(self, vision_info_list, conversation):
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten()
        new_input_ids = input_ids
        vision_start_token_id = getattr(self.config, "vision_start_token_id", _IMAGE_START_ID)
        vision_end_token_id = getattr(self.config, "vision_end_token_id", _IMAGE_END_ID)

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
