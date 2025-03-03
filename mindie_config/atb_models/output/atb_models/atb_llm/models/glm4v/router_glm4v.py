# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import importlib
from typing import Dict, List
from dataclasses import dataclass
import os
import numpy as np
import torch

from atb_llm.utils.shm_utils import create_shm, encode_shm_name_to_int64, encode_shape_to_int64
from ..base.router import BaseRouter
from .input_builder_glm4v import Glm4vInputBuilder
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode
from ...utils.multimodal_utils import safe_open_image

_PAD_TOKEN_ID = 151329
_BOI_TOKEN_ID = 151339
_EOI_TOKEN_ID = 151340
_IMG_TOKEN_LEN = 1600


@dataclass
class Glm4vRouter(BaseRouter):
    def tokenize(self, inputs: List[Dict], **kwargs):
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        boi_token_id = self.config.boi_token_id if self.config.boi_token_id is not None else _BOI_TOKEN_ID
        eoi_token_id = self.config.eoi_token_id if self.config.eoi_token_id is not None else _EOI_TOKEN_ID

        image_path = ""
        text = ""
        for ele in inputs:
            if "image_or_video" in ele:
                image_path = ele["image_or_video"]
            elif "image" in ele:
                image_path = ele["image"]
            elif "text" in ele:
                text = ele["text"]
        if image_path == "" or text == "":
            logger.error("error: image path or text path is empty, please use legal path",
                         ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError("image path or text is empty.")
        
        # image preprocess
        from PIL import Image
        image_pix = safe_open_image(Image, image_path).convert("RGB")
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image_pix, "content": text}],
                                               add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)
        input_ids = inputs["input_ids"].flatten()
        input_image = inputs["images"]

        # save image to shared memory
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        if shm_name_save_path is None:
            shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
            shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
        shm = create_shm(input_image.nbytes, shm_name_save_path)
        shared_array = np.ndarray(input_image.shape, dtype=np.float32, buffer=shm.buf)
        shared_array[:] = input_image
        # encode shared memory name and image's shape
        shm_name = encode_shm_name_to_int64(shm.name)
        shape_value = encode_shape_to_int64(input_image.shape)

        # prepare input_ids with encoded shm info
        img_padding = torch.full((_IMG_TOKEN_LEN, ), pad_token_id, dtype=input_ids.dtype)
        new_input_ids = []
        boi_pos = torch.where(torch.eq(input_ids, boi_token_id))[0]
        eoi_pos = torch.where(torch.eq(input_ids, eoi_token_id))[0]
        new_input_ids.append(torch.cat((input_ids[:boi_pos + 1], img_padding, input_ids[eoi_pos:])))
        new_input_ids = torch.cat(new_input_ids)
        new_input_ids[boi_pos + 1] = shm_name
        new_input_ids[boi_pos + 2] = shape_value
        return new_input_ids
    
    def get_input_builder(self):
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            return Glm4vInputBuilder(self.tokenizer, self.config, max_length=self.config.max_position_embeddings)
        return Glm4vInputBuilder(self.tokenizer, self.config)
    
    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        config.model_name_or_path = self.model_name_or_path
        self.check_config(config)
        return config
    
    def check_config(self, config):
        vocab_size = 0
        vocab_size_string = "vocab_size"
        padded_vocab_size_string = "padded_vocab_size"
        if hasattr(config, vocab_size_string):
            vocab_size = getattr(config, vocab_size_string)
        if hasattr(config, padded_vocab_size_string):
            vocab_size = getattr(config, padded_vocab_size_string)
        attribute_ranges = {
            'kv_channels': (1, 2147483647),
            'multi_query_group_num': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
            'layernorm_epsilon': (0, 1),
            'attention_dropout': (0, 1),
            'ffn_hidden_size': (1, 2147483647),
            'hidden_dropout': (0, 1),
            padded_vocab_size_string: (1, 2147483647),
            'rope_ratio': (0, 2147483647),
            'seq_length': (1, 2147483647),
            vocab_size_string: (1, 2147483647),
            'max_position_embeddings': (1, 2147483647),
            'hidden_size': (1, 2147483647),
            'intermediate_size': (1, 2147483647),
            'num_hidden_layers': (1, 1000),
            'num_attention_heads': (1, 10000),
            'initializer_range': (0, 2147483647),
            'rms_norm_eps': (0, 1),
            'pad_token_id': (0, vocab_size),
            'bos_token_id': (0, vocab_size),
            'eos_token_id': (0, vocab_size)
        }
        if hasattr(config, "head_dim"):
            attribute_ranges['head_dim'] = (1, 1000)
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if isinstance(value, list):
                value = max(value)
            if value < min_val or value > max_val:
                logger.error(f"Error: self._config.{attr} is out of range", 
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_config_cls(self):
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        config_file_name = 'config'
        module_path = f"{model_file_dir_name}{config_file_name}_{self.model_type}"
        module = importlib.import_module(module_path)
        config_cls_name = f"{self.model_type_cap}Config"
        return getattr(module, config_cls_name)

    def get_model_cls(self):
        """
        get_model_cls
        """
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self.model_type_cap}ForCausalLM"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        return getattr(module, model_cls_name)
