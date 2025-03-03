# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""PyTorch Yivl model."""

import re
import torch
import torch.nn as nn
import numpy as np
from atb_llm.utils.shm_utils import get_data_from_shm
from atb_llm.models.base.flash_causal_multimodal import MultiModalLLm, get_llm_model
from atb_llm.models.yivl.input_builder_yivl import tokenize_text, IMAGE_TOKEN_INDEX
from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode

    
class YivlConfig(BaseConfig):
    audio_config = None

    @property
    def text_config(self):
        return self._text_config

    @text_config.setter
    def text_config(self, value):
        self._text_config = value

    @property
    def vision_config(self):
        return self._vision_config

    @vision_config.setter
    def vision_config(self, value):
        self._vision_config = value


def get_multimodal_projector(config: YivlConfig):
    use_norm = False    
    mlp_depth = 1

    projector_type = config.mm_projector_type
    if "_Norm" in projector_type:
        use_norm = True
    projector_type = projector_type.replace("_Norm", "")
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))

    if use_norm:
        modules = [
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        ]
    else:
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        if use_norm:
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            modules.append(nn.LayerNorm(config.hidden_size))
        else:
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))

    return nn.Sequential(*modules)


class FlashYivlForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        kwargs.update({"lmhead_prefix": "lm_head"})
        kwargs.update({"model_prefix": "model"})
        kwargs.update({"vision_prefix": "model.vision_tower.vision_tower"})
        super().__init__(config, weights, **kwargs)
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self._init_multimodal()

    def prepare_prefill_token(self, multimodalparams, processor):
        image = multimodalparams.image
        text = multimodalparams.text
        pixel_values = processor.image_processor.preprocess_image(self.config, image)
        image_features = self._extract_image_feature(pixel_values)

        input_ids = tokenize_text(text, processor.tokenizer)
        inputs_embeds = self._merge_input_ids_with_image_features(image_features, input_ids)
        return inputs_embeds
    
    def prepare_prefill_token_service(self, input_ids):
        inputs_embeds = None
        img_token_id = self.config.img_token_id
        if torch.any(torch.eq(input_ids, img_token_id)):
            img_token_pos = torch.where(torch.eq(input_ids, img_token_id))[0]
            image_num = img_token_pos.size(0) // 2
            for i in range(0, image_num * 2, 2):
                shm_value = input_ids[img_token_pos[i] + 1]
                shape_value = input_ids[img_token_pos[i] + 2]
                pixel_values = get_data_from_shm(shm_value, shape_value, np.float32, self.device)

                image_features = self._extract_image_feature(pixel_values)
                inputs_embeds = self._merge_input_ids_with_image_features_service(image_features, 
                                                                                  input_ids,
                                                                                  img_token_pos)

        return inputs_embeds if inputs_embeds is not None else self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def init_llm(self):
        model_cls = get_llm_model("llama")
        self.language_model = model_cls(self.config.text_config,
                                        self.weights,
                                        self.lmhead_prefix,
                                        self.model_prefix)
        self.language_model.skip_word_embedding = True
    
    def _init_multimodal(self):
        self.multi_modal_projector = get_multimodal_projector(self.config)
        self.init_tower_weight(self.multi_modal_projector, self.weights, "model.mm_projector")
        
    def _merge_input_ids_with_image_features(self, image_features, input_ids):
        image_token_pos = input_ids.index(IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor(input_ids)
        image_features = image_features.squeeze(0)

        head_embeds = self.get_input_embeddings()(input_ids[: image_token_pos].npu()) 
        tail_embeds = self.get_input_embeddings()(input_ids[image_token_pos + 1 :].npu())
        return torch.cat([head_embeds, image_features, tail_embeds], dim=-2) # -2 seq dim

    def _merge_input_ids_with_image_features_service(self, image_features, input_ids, img_token_pos):
        image_features = image_features.squeeze(0)
        head_embeds = self.get_input_embeddings()(input_ids[:img_token_pos[0]]).npu()
        tail_embeds = self.get_input_embeddings()(input_ids[img_token_pos[1] + 1:]).npu()
        return torch.cat([head_embeds, image_features, tail_embeds], dim=-2) # -2 seq dim
    
    def _extract_image_feature(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values.to(dtype=self.dtype).npu(), output_hidden_states=True)
        vision_feature_layer = self.config.mm_vision_select_layer
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        vision_feature_select_strategy = self.config.mm_vision_select_feature

        if vision_feature_select_strategy == "patch":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.select_feature == "cls_patch":
            selected_image_feature = selected_image_feature
        else:
            msg = f"Unexpected select feature: {self.select_feature}"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        image_features = self.multi_modal_projector(selected_image_feature)

        return image_features
