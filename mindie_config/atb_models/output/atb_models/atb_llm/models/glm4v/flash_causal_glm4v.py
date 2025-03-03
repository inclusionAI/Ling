# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 THUDM and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on HuggingFace's glm-4v-9b developed by THUDM.
# It has been modified from its original forms to accommodate minor 
# architectural differences compared to glm-4V-9b.
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

from typing import Optional, List, Tuple
import importlib
import torch
import numpy as np
from PIL import Image

from transformers.configuration_utils import PretrainedConfig
from atb_llm.utils.shm_utils import get_data_from_shm
from ..base.flash_causal_lm import FlashForCausalLM
from .modeling_glm_vit import EVA2CLIPModel
from ...utils.multimodal_utils import safe_open_image

_GMASK_TOKEN_ID = 151331
_PLACEHOLDER = 0


class FlashGlm4vForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.vocab_size
        self.vision = None
        self.language_model = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.init_vit()
        self.init_llm()
        self.model_type = config.model_type

    @staticmethod
    def init_vision_weight(module, weights, prefix):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"{prefix}.{model_weight}"),
                requires_grad=False
            )
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    @staticmethod
    def get_llm_model(model_type):
        model_file_dir_name = f"atb_llm.models.{model_type}."
        model_file_name = "flash_causal"
        module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
        return getattr(module, model_cls_name)
    
    def init_vit(self):
        self.vision = EVA2CLIPModel(self.config)
        self.init_vision_weight(self.vision, self.weights, "transformer.vision")

    def init_llm(self):
        model_cls = self.get_llm_model(self.config.llm_model_type)
        self.language_model = model_cls(self.config, self.weights)
        self.language_model.skip_word_embedding = True

    def prepare_prefill_token_service(self, input_ids):
        if not torch.any(torch.eq(input_ids, self.config.boi_token_id)):
            return self.language_model.embedding.word_embeddings(input_ids)

        inputs_embeds = self.language_model.embedding.word_embeddings(input_ids)
        
        batch_boi_pos = torch.where(torch.eq(input_ids, self.config.boi_token_id))[0]
        batch_eoi_pos = torch.where(torch.eq(input_ids, self.config.eoi_token_id))[0]
        for idx, boi_pos in enumerate(batch_boi_pos):
            eoi_pos = batch_eoi_pos[idx]
            # get shm info from input_ids
            shm_value = input_ids[boi_pos + 1]
            shape_value = input_ids[boi_pos + 2]
            # get image feature
            input_image = get_data_from_shm(
                shm_value, shape_value, np.float32, self.device
            ).to(dtype=inputs_embeds.dtype).npu()
            image_features = self.vision(input_image)

            # replace embeds with image feature
            inputs_embeds[boi_pos:eoi_pos + 1] = image_features
        return inputs_embeds

    def prepare_prefill_token(self, multimodalinputs, processor, *args):
        image = multimodalinputs.image
        text = multimodalinputs.text

        image = safe_open_image(Image, image).convert("RGB")
        inputs = processor.apply_chat_template([{"role": "user", "image": image, "content": text}],
                                               add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)
        input_ids = inputs["input_ids"].npu()
        inputs_embeds = self.language_model.embedding.word_embeddings(input_ids)

        input_image = inputs["images"].to(dtype=inputs_embeds.dtype).npu()
        image.close()
        images_features = self.vision(input_image)

        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        num_patches = (image_size // patch_size // 2) ** 2

        new_input_embeds, new_position_ids = [], []
        position_ids = torch.arange(len(input_ids[0]), dtype=torch.long).unsqueeze(0).repeat(1, 1)
        for i, x in enumerate(input_ids):
            input_id = x.tolist()
            boi_token_pos = input_id.index(self.config.boi_token_id)
            eoi_token_pos = input_id.index(self.config.eoi_token_id)
            new_input_embeds.append(torch.cat(
                (inputs_embeds[i, :boi_token_pos], images_features[i].to(inputs_embeds.device),
                    inputs_embeds[i, eoi_token_pos + 1:])))
            new_position_ids.append(torch.cat(
                (position_ids[i, :boi_token_pos + 1], position_ids[i, boi_token_pos + 1].repeat(num_patches),
                    position_ids[i, eoi_token_pos:])
            ))
        
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        position_ids = torch.stack(new_position_ids, dim=0)
        return inputs_embeds.squeeze(0), position_ids.squeeze(0)
    
    def init_ascend_operation(self, config: PretrainedConfig):
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
                                           lm_head_indices,
                                           **kwargs)