# Tongyi Qianwen is licensed under the Tongyi Qianwen LICENSE AGREEMENT, 
# Copyright (c) Alibaba Cloud. All Rights Reserved.
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from typing import Dict, List
import torch
import torch_npu
import numpy as np

from atb_llm.utils import shm_utils
from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.models.qwen.vl.data_preprocess_qwen_vl import qwen_vl_image_preprocess
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode

_ERROR_BAD_CHAT_FORMAT = """\
We have detected that you might be using a pretrained moedl for chatting instead of a
delicated chat model. Please make sure that you are using the correct model. For example,
make sure you are using `Qwen/Qwen-XXX-Chat` instead of `Qwen/Qwen-XXX`.
我们检测到您可能在使用预训练模型进行对话，而不是专门的chat模型，请确认您使用的是正确的模型。
例如，请确认您使用的是`Qwen/Qwen-XXX-Chat`而不是`Qwen/Qwen-XXX`。
"""


class QwenVLInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, generation_config, image_start_id, **kwargs):
        self.model_version = model_version
        self.generation_config = generation_config
        self.image_start_id = image_start_id
        super().__init__(tokenizer, system_role_name="assistant", user_role_name="user", **kwargs)

    def make_context(
        self, 
        rank: int,
        conversation: List[Dict[str, List[Dict]]], 
        system: str = "You are a helpful assistant.",
        **kwargs):
        if self.generation_config["chat_format"] != 'chatml':
            logger.error(_ERROR_BAD_CHAT_FORMAT, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(_ERROR_BAD_CHAT_FORMAT)
        if not isinstance(conversation[0]["content"], list):
            msg = "The conversation \"content\" should be a List[Dict]."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        self.rank = rank
        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = self.generation_config["max_window_size"]

        context_tokens = self._apply_chat_template(
            conversation,
            system=system,
            max_window_size=max_window_size,
            shm_name_save_path=shm_name_save_path,
            )
        return context_tokens

    def _apply_chat_template(
        self,
        conversation: List[Dict[str, List[Dict]]],
        system: str = "",
        max_window_size: int = 6144,
        shm_name_save_path: str = None,
        **kwargs):
        
        im_start_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        nl_tokens = self.tokenizer.encode("\n")

        system_tokens_part = self._tokenize_str("system", system, nl_tokens)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        shm_name_list = []
        shape_value_list = []
        content_key = "content"
        image_key = "image"
        for message in conversation:
            for single_input in message[content_key]:
                if image_key not in single_input.keys():
                    continue
                image_pixel = qwen_vl_image_preprocess(single_input[image_key])
                image_pixel = image_pixel[None, :]
                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[image_key]))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
                shm = shm_utils.create_shm(image_pixel.nbytes, shm_name_save_path)
                shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
                shared_array[:] = image_pixel

                shm_name = shm_utils.encode_shm_name_to_int64(shm.name)
                shape_value = shm_utils.encode_shape_to_int64(image_pixel.shape)
                shm_name_list.append(shm_name)
                shape_value_list.append(shape_value)

        context_tokens = system_tokens
        query = self.tokenizer.from_list_format(conversation.pop()[content_key])

        for message in conversation[::-1]:
            turn_query = self.tokenizer.from_list_format(message[content_key])
            if message["role"] == self.user_role_name:
                query_tokens = nl_tokens + im_start_tokens + \
                    self._tokenize_str(self.user_role_name, turn_query, nl_tokens) + im_end_tokens + nl_tokens
            elif message["role"] == self.system_role_name:
                query_tokens = im_start_tokens + \
                    self._tokenize_str(self.system_role_name, turn_query, nl_tokens) + im_end_tokens
            else:
                msg = f"message 'role' not supported yet"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)

            current_context_size = (
                len(system_tokens) + len(query_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = query_tokens + context_tokens
            else:
                break
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + self._tokenize_str(self.user_role_name, query, nl_tokens)
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + self.tokenizer.encode(self.system_role_name)
            + nl_tokens
        )

        context_tokens_tensor = torch.tensor(context_tokens)
        bos_pos = torch.where(torch.eq(context_tokens_tensor, self.image_start_id))[0]
        image_num = bos_pos.shape[0]
        for i in range(image_num):
            context_tokens[bos_pos[i] + 1] = shm_name_list[i]
            context_tokens[bos_pos[i] + 2] = shape_value_list[i]

        return context_tokens

    def _tokenize_str(self, role, content, nl_tokens):
        return self.tokenizer.encode(
            role, allowed_special=set(self.tokenizer.IMAGE_ST)
        ) + nl_tokens + self.tokenizer.encode(content, allowed_special=set(self.tokenizer.IMAGE_ST))