# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from typing import Dict, List
import os
import torch
import numpy as np
from transformers import PretrainedConfig, CLIPVisionConfig
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter
from ..llama.config_llama import LlamaConfig
from ..yivl.flash_causal_yivl import YivlConfig
from ..base.config import QuantizationConfig
from .input_builder_yivl import SEP_TOKEN, IMAGE_TOKEN_INDEX
from .input_builder_yivl import tokenize_text
from .data_processor_yivl import DataProcessorYiVl
from ..base.model_utils import safe_from_pretrained, safe_get_tokenizer_from_pretrained

_PAD_TOKEN_ID = 0
_EOS_TOKEN_ID = 8308
_INT32_MAX = 2147483647


def check_value_range(attribute_ranges, config):
    for attr, (min_val, max_val) in attribute_ranges.items():
        if not hasattr(config, attr) or getattr(config, attr) is None:
            continue
        value = getattr(config, attr)
        if value < min_val or value > max_val:
            msg = f"self._config.{attr} must be between {min_val} and {max_val}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)


def parse_service_inputs(inputs : List[Dict]):
    image_exist_flag = False
    image = None
    text = ""
    for ele in inputs:
        if "image" in ele:
            if not image_exist_flag:
                image = ele["image"]
                image_exist_flag = True
            else:
                logger.warning("Yi-vl only support only one image at once, others will be ignored!")
        elif "text" in ele:
            text += ele["text"]
        else:
            msg = "Unsupport element: " + str(ele)
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise KeyError(msg)
    if not image_exist_flag:
        logger.info("No image found in request!")
    return image, text


@dataclass
class YivlRouter(BaseRouter):
    model_type = "yivl"

    def __post_init__(self):
        super().__post_init__()
        self.image_processor = DataProcessorYiVl(self.config.vison_tower_path,
                                            self.trust_remote_code)

    def check_config_yivl(self, config):
        super().check_config(config)
        vision_attribute_ranges = {
            'hidden_size': (1, _INT32_MAX),
            'image_size': (1, _INT32_MAX),
            'intermediate_size': (1, _INT32_MAX),
            'num_attention_heads': (1, _INT32_MAX),
            'num_channels': (1, _INT32_MAX),
            'num_hidden_layers': (1, _INT32_MAX),
            'patch_size': (1, config.vision_config.image_size),
            'projection_dim': (1, _INT32_MAX)
        }
        check_value_range(vision_attribute_ranges, config.vision_config)

        vision_layers = config.vision_config.num_hidden_layers
        attribute_ranges = {
            'mm_hidden_size': (1, _INT32_MAX),
            'num_key_value_heads': (1, _INT32_MAX),
            'mm_vision_select_layer': (-vision_layers, vision_layers),
        }
        check_value_range(attribute_ranges, config)

    def get_config(self) -> PretrainedConfig:
        config = YivlConfig.from_pretrained(self.model_name_or_path)
        config.model_type = self.model_type

        if config.mm_vision_tower:
            from os.path import join
            config.vison_tower_path = join(
                self.model_name_or_path, config.mm_vision_tower.replace("./", "")
            )
            config.vision_config = safe_from_pretrained(CLIPVisionConfig, config.vison_tower_path)
        else:
            msg = "key 'mm_vision_tower' not found at config"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        config.text_config = LlamaConfig.from_pretrained(self.model_name_or_path)

        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        config.vocab_size = config.text_config.vocab_size
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        self.check_config_yivl(config)
        config.model_name_or_path = self.model_name_or_path
        setattr(config, 'img_token_id', IMAGE_TOKEN_INDEX)
        setattr(config, 'eos_token_id', getattr(self.tokenizer, 'eos_token_id', _EOS_TOKEN_ID))
        setattr(config, 'pad_token_id', getattr(self.tokenizer, 'pad_token_id', _PAD_TOKEN_ID))
        setattr(config, 'num_img_patches', (config.vision_config.image_size // config.vision_config.patch_size) ** 2)
        return config
    
    def get_tokenizer(self):
        use_fast = True
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        tokenizer.eos_token = SEP_TOKEN
        return tokenizer


    def get_generation_config(self):
        generation_config = super().get_generation_config()
        generation_config["eos_token_id"] = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        return generation_config

    def tokenize(self, inputs, **kwargs):
        img_token_id = self.config.img_token_id
        pad_token_id = self.config.pad_token_id
        img_patch_num = self.config.num_img_patches
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        image_path, question = parse_service_inputs(inputs)
        input_ids = tokenize_text(question, self.tokenizer)

        new_input_ids = input_ids
        if image_path:
            image_pixel = self.image_processor.preprocess_image(self.config, image_path)
            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            shm = create_shm(image_pixel.nbytes, shm_name_save_path)
            shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
            shared_array[:] = image_pixel
            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(image_pixel.shape)
    
            image_info_ids = [img_token_id, shm_name, shape_value]
            if (img_patch_num < len(image_info_ids) + 1):
                msg = "image patch num is too small!"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
            pad_num = img_patch_num - 1 - len(image_info_ids)
            image_info_ids.extend([pad_token_id] * pad_num)

            image_pos = input_ids.index(img_token_id)
            new_input_ids = input_ids[:image_pos] + image_info_ids + input_ids[image_pos:]

        return torch.tensor(new_input_ids)