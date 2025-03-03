# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Llava model."""

from typing import Optional, List, Tuple
import abc
import importlib
import os
import warnings

import torch
from torch import nn
from PIL import Image
from transformers.activations import ACT2FN
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import AutoModel
from transformers import AutoProcessor

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.flash_causal_lm import FlashForCausalLM
from ..base.config import QuantizationConfig, BaseConfig
from ..base.model_utils import safe_from_pretrained
from ...utils.multimodal_utils import safe_open_image

MODEL_TYPE = "model_type"
LLAMA = "llama"
MISTRAL = "mistral"
VICUNA = "vicuna"
_PAD_TOKEN_ID = 32001


def is_video(file_name):
    ext = os.path.splitext(file_name)[1]
    ext = ext.lower()
    if ext in [".mp4", ".wmv", ".avi"]:
        return True
    return False


def get_supported_models():
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    supported_models = []
    for foldername in os.listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)
    return supported_models


def get_llm_model(model_type):
    supported_models = get_supported_models()
    if model_type not in supported_models:
        msg = "Unsupported model type." + \
              " Verify that the corresponding folder exists in the atb_llm.models path."
        logger.error(
            msg,
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
        )
        raise NotImplementedError(msg)
    
    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name = 'flash_causal'
    module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
    module = importlib.import_module(module_path)
    model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class MultiModalConfig(BaseConfig):
    model_type = "llava"
    is_composition = False

    def __init__(self, vision_config=None, text_config=None, **kwargs):
        
        self._init_visionconfig(vision_config)
        self._init_textconfig(text_config)
        super().__init__(**kwargs)
    
    def _init_visionconfig(self, vision_config):
        if isinstance(vision_config, dict):
            vision_config[MODEL_TYPE] = (
                vision_config[MODEL_TYPE] if MODEL_TYPE in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config[MODEL_TYPE]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
        self.vision_config = vision_config

    def _init_textconfig(self, text_config):
        self.text_config = text_config
        

class LlavaConfig(MultiModalConfig):

    model_type = "llava"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        **kwargs,
    ):
        super().__init__(vision_config, text_config, **kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.max_position_embeddings = 4096

        if vision_feature_select_strategy not in ["default", "full"]:
            msg = "vision_feature_select_strategy should be one of 'default', 'full'."
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)

        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, \
                since it can be inferred from the `text_config`. \
                Passing this argument has no effect",
                FutureWarning,
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self._vocab_size = self.text_config["vocab_size"] if self.text_config is not None else 0 

    @property
    def vocab_size(self):
        warnings.warn(
            "The `vocab_size` attribute is deprecated and will be removed in v4.42, \
            Please use `text_config.vocab_size` instead.",
            FutureWarning,
        )
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        return output


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MultiModalLLm(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if getattr(config, "text_config"):
            if not config.quantize:
                setattr(config.text_config, 'quantize', None)
            else:
                setattr(config.text_config, 'quantize', config.quantize)
            setattr(config.text_config, 'quantization_config', QuantizationConfig(**{}))
            super().__init__(config.text_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.text_config.vocab_size
        self.vision_tower = None
        self.language_model = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.init_vit()
        self.init_llm()
        self.model_type = None

    @staticmethod
    def init_visiontowerweight(module, weights):
        vision_weights = [vision_weight for vision_weight in module.state_dict().keys()]
        for vision_weight in vision_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"vision_tower.{vision_weight}"),
                    requires_grad=False
                )
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    def init_vit(self):
        self.vision_tower = AutoModel.from_config(self.config.vision_config)
        self.init_visiontowerweight(self.vision_tower, self.weights)

    def init_llm(self):
        self.model_type = self.config.text_config.model_type
        if self.model_type in [MISTRAL, VICUNA, LLAMA]:
            self.model_type = LLAMA
        model_cls = get_llm_model(self.model_type)
        self.language_model = model_cls(self.config.text_config, 
                                  self.weights,
                                  "language_model.lm_head",
                                  "language_model.model")
        self.language_model.skip_word_embedding = True

    @abc.abstractmethod
    def prepare_prefill_token_service(self, input_ids):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs):
        return self.language_model.forward(input_ids, 
                                          position_ids,
                                          is_prefill,
                                          kv_cache,
                                          block_tables,
                                          slots,
                                          input_lengths,
                                          max_seq_len,
                                          lm_head_indices)


class FlashLlavaForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        self.multi_modal_projector = None
        self.image_token_id = self.config.image_token_index
        self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path, 
                                              trust_remote_code=self.trust_remote_code)
        self.init_multimodal()
        
    @staticmethod
    def init_multi_modal_projectorweight(module, weights):
        multimodel_weights = [multimodel_weight for multimodel_weight in module.state_dict().keys()]
        for multimodel_weight in multimodel_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"multi_modal_projector.{multimodel_weight}"),
                    requires_grad=False
                )
            multimodel_weight_list = multimodel_weight.split(".")
            target_module = module
            for nxt_module in multimodel_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, multimodel_weight_list[-1], saved_weight)

    def init_multimodal(self):
        self.multi_modal_projector = LlavaMultiModalProjector(self.config)
        self.init_multi_modal_projectorweight(self.multi_modal_projector, self.weights)
    
    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def prepare_prefill_token_service(self, input_ids):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if torch.any(torch.eq(input_ids, self.image_token_id)):
            bos_pos = torch.where(torch.eq(input_ids, self.image_token_id))[0]
            image_num = bos_pos.size(0) // 2
            for i in range(0, image_num * 2, 2):
                path_token = input_ids[bos_pos[i]: bos_pos[i + 1] + 1]
                image_path_mask = (path_token != self.image_token_id) & (path_token != self.pad_token_id)
                image_path = self.processor.batch_decode(path_token[image_path_mask].unsqueeze(0))
                if is_video(image_path[0]):
                    msg = "Llava 1.5 does not support video input"
                    logger.error(
                        msg,
                        ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
                    )
                    raise RuntimeError(msg)
                image = safe_open_image(Image, image_path[0])
                pixel_values = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
                image.close()
                image_outputs = self.vision_tower(pixel_values.half().npu(), output_hidden_states=True)
                vision_feature_layer = self.config.vision_feature_layer
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
                vision_feature_select_strategy = self.config.vision_feature_select_strategy
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                image_features = self.multi_modal_projector(selected_image_feature)  
                inputs_embeds = self._merge_input_ids_with_image_features_service(image_features,
                                                                        inputs_embeds, 
                                                                        bos_pos[i: i + 2]) 
        return inputs_embeds

    def prepare_prefill_token(self, multimodalinputs, processor):
        text = multimodalinputs.text
        image = multimodalinputs.image
        video = multimodalinputs.video
        if video:
            msg = "Llava 1.5 does not support video input"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise RuntimeError(msg)
        image = safe_open_image(Image, image)
        pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        image.close()
        image_outputs = self.vision_tower(pixel_values.half().npu(), output_hidden_states=True)
        vision_feature_layer = self.config.vision_feature_layer
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        selected_image_feature = selected_image_feature
        image_size = selected_image_feature.shape[1]
        image_token_str = processor.tokenizer.decode(self.config.image_token_index)
        replacement_str = "".join([image_token_str] * image_size)
        new_prompt = text.replace(image_token_str, replacement_str, 1)
        input_ids = processor.tokenizer(new_prompt, return_tensors="pt")["input_ids"].npu()
        image_features = self.multi_modal_projector(selected_image_feature)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = self._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds, 
            input_ids,
            self.config.image_token_index
        )
        inputs_embeds = inputs_embeds.view(inputs_embeds.shape[0] * inputs_embeds.shape[1],
                                            inputs_embeds.shape[2])
        return inputs_embeds

    def init_ascend_operations(self, config: BaseConfig):
        pass

    def init_ascend_weight(self):
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs):
        if is_prefill and input_ids.dim() == 1:
            input_ids = self.prepare_prefill_token_service(input_ids)
        return self.language_model.forward(input_ids, 
                                          position_ids,
                                          is_prefill,
                                          kv_cache,
                                          block_tables,
                                          slots,
                                          input_lengths,
                                          max_seq_len,
                                          lm_head_indices)
    
    def _merge_input_ids_with_image_features_service(self, image_features, inputs_embeds, bos_pos):
        inputs_embeds[bos_pos[0]: bos_pos[1] + 1] = image_features
        return inputs_embeds
    
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, image_token_id):
        mask = (input_ids == image_token_id)
        inputs_embeds[mask] = image_features
        return inputs_embeds