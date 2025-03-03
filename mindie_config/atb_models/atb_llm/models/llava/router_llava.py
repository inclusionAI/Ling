# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from typing import Dict, List

import torch

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter
from ..llava.flash_causal_llava import LlavaConfig
from ..llama.config_llama import LlamaConfig
from ..base.config import QuantizationConfig
from ..base.model_utils import safe_get_tokenizer_from_pretrained

_IMAGE_TOKEN_ID = 32000
_PAD_TOKEN_ID = 32001
_IMAGE_FEATURE_WIDTH = 576


def from_list_format(list_format: List[Dict], image_token_str: str):
    text = "User: "
    for ele in list_format:
        if "image_or_video" in ele:
            text += image_token_str + ele["image_or_video"] + image_token_str
        elif "image" in ele:
            text += image_token_str + ele["image"] + image_token_str
        elif "text" in ele:
            text += ele["text"]
        else:
            msg = "Unsupport element: " + str(ele)
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise KeyError(msg)
    return text


@dataclass
class LlavaRouter(BaseRouter):

    def tokenize(self, inputs, **kwargs):
        pad_token_id = self.config.pad_token_id \
            if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        image_token_id = self.config.image_token_index \
            if self.config.image_token_index is not None else _IMAGE_TOKEN_ID
        image_token_str = self.tokenizer.decode(image_token_id)
        prompt = from_list_format(inputs, image_token_str)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten()
        new_input_ids = input_ids
        bos_pos = torch.where(torch.eq(input_ids, image_token_id))[0]  
        if bos_pos.size(0) % 2 != 0:
            msg = "Ensure that your input_text does not contain the '<image>' character!"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        image_num = bos_pos.size(0) // 2
        expand_token_ids = []
        pre = 0
        for i in range(0, image_num * 2, 2):
            new_path_token = torch.tensor([])
            path_token_len = bos_pos[i + 1] - bos_pos[i]
            if path_token_len < 0 or path_token_len >= _IMAGE_FEATURE_WIDTH:
                msg = f"Token length in path must be greater than 0 and less than {_IMAGE_FEATURE_WIDTH}"
                logger.error(
                    msg,
                    ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
                )
                raise ValueError(msg)
            text_token = input_ids[pre: bos_pos[i]]
            path_token = input_ids[bos_pos[i] + 1: bos_pos[i + 1] + 1]
            pre = bos_pos[i + 1] + 1
            new_path_token = torch.cat(
                [
                    torch.tensor([image_token_id], dtype=path_token.dtype),
                    torch.full((_IMAGE_FEATURE_WIDTH - path_token_len - 1, ), pad_token_id, dtype=path_token.dtype),
                    path_token
                ]
            )
            if text_token.size(0) != 0:
                expand_token_ids.append(text_token)
                
            if new_path_token.size(0) != 0:
                expand_token_ids.append(new_path_token)
                
        text_token = input_ids[pre:]
        if text_token.size(0) != 0:
            expand_token_ids.append(text_token)
            
        if expand_token_ids:
            new_input_ids = torch.cat(expand_token_ids)
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
                msg = f"The {attr} value in config must be between {min_val} and {max_val}"
                logger.error(
                    msg,
                    ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
                )
                raise ValueError(msg)

    def get_config(self):
        config = LlavaConfig.from_pretrained(self.model_name_or_path)
        config.text_config = LlamaConfig.from_dict(config.text_config)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_llava(config)
        config.model_name_or_path = self.model_name_or_path
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )
