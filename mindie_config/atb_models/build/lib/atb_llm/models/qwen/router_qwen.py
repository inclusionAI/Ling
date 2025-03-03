# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
import os
import numpy as np
import torch

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.qwen.config_qwen import QwenConfig
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .vl.input_builder_qwen_vl import QwenVLInputBuilder
from .vl.data_preprocess_qwen_vl import qwen_vl_image_preprocess


@dataclass
class QwenRouter(BaseRouter):
    jinja_model_template = """
        {%- for message in messages -%}
            {%- if loop.first and messages[0]['role'] != 'system' -%}
                {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' -}}
            {%- endif -%}
            {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' -}}
        {%- endfor -%}

        {%- if add_generation_prompt -%}
            {{- '<|im_start|>assistant\n' -}}
        {%- endif -%}
        """

    @property
    def model_version(self):
        """
        次级模型:主要用于区分qwen-vl
        """
        if "visual" in self.config_dict:
            model_ver = "vl"
        else:
            model_ver = ""
        return model_ver

    def tokenize(self, inputs, **kwargs):
        query = self.tokenizer.from_list_format(inputs)
        input_ids = self.tokenizer([query], return_tensors="pt")["input_ids"].flatten()
        shm_name_save_path = kwargs.get('shm_name_save_path', None)

        shm_name_list = []
        shape_value_list = []
        image_type = "image"
        for single_input in inputs:
            if image_type not in single_input.keys():
                continue
            image_pixel = qwen_vl_image_preprocess(single_input[image_type])
            image_pixel = image_pixel[None, :]
            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[image_type]))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            shm = create_shm(image_pixel.nbytes, shm_name_save_path)
            shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
            shared_array[:] = image_pixel

            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(image_pixel.shape)
            shm_name_list.append(shm_name)
            shape_value_list.append(shape_value)

        image_start_id = self.config.visual["image_start_id"]
        bos_pos = torch.where(torch.eq(input_ids, image_start_id))[0]
        image_num = bos_pos.shape[0]
        for i in range(image_num):
            if input_ids.size(0) < bos_pos[i] + 3:
                msg = "tokenize error, input_ids length is too short."
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
            input_ids[bos_pos[i] + 1] = shm_name_list[i]
            input_ids[bos_pos[i] + 2] = shape_value_list[i]

        return input_ids

    def get_config(self):
        config = QwenConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
        )
        self.checkout_config_qwen(config)       
        return config

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=self.trust_remote_code,
        )

    def get_input_builder(self):
        if self.model_version == "vl":
            return QwenVLInputBuilder(self.tokenizer, self.model_version, self.generation_config, 
                self.config.visual["image_start_id"])
        
        max_length = self.max_position_embeddings
        if hasattr(self.generation_config, "max_length"):
            max_length = self.generation_config.max_length
        if hasattr(self.config, "chat_template"):
            return InputBuilder(self.tokenizer, self.config.chat_template, max_length=max_length)
        else:
            return InputBuilder(self.tokenizer, self.jinja_model_template, max_length=max_length)

    def checkout_config_qwen(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attention_dropout': (0, 2147483647),
            'emb_dropout_prob': (0, 2147483647), 
            'kv_channels': (1, 2147483647),
            'layer_norm_epsilon': (0, 1),
            'rotary_emb_base': (1, 2147483647),
            'rotary_pct': (1, 2147483647),
            'seq_length': (1, 2147483647)
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr):
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
