# coding=utf-8
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# --------------------------------------------------------
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from dataclasses import dataclass

from atb_llm.models.base.config import BaseConfig
from atb_llm.models.internvl.config_intern_vit import InternVisionConfig
from atb_llm.models.internlm2.v2.config_internlm2 import Internlm2Config
from atb_llm.models.llama.config_llama import LlamaConfig
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger


@dataclass
class InternvlConfig(BaseConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(self,
                 vision_config=None,
                 llm_config=None,
                 use_backbone_lora=0,
                 use_llm_lora=0,
                 select_layer=-1,
                 force_image_size=None,
                 downsample_ratio=0.5,
                 template=None,
                 dynamic_image_size=False,
                 use_thumbnail=False,
                 ps_version='v1',
                 min_dynamic_patch=1,
                 max_dynamic_patch=12,
                 **kwargs):
        llm_config["quantize"] = None
        llm_config["quantization_config"] = None
        super().__init__(**llm_config)

        self.vision_config = InternVisionConfig(**vision_config)
        llm_model_architectures = llm_config['architectures'][0]
        if llm_model_architectures == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_model_architectures == 'InternLM2ForCausalLM':
            self.llm_config = Internlm2Config(**llm_config)
        else:
            logger.error('Unsupported architecture: {}'.format(llm_model_architectures),
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError('Unsupported architecture: {}'.format(llm_model_architectures))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
