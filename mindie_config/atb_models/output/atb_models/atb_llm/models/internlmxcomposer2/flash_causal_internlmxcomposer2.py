# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
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
# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
import abc
import importlib
import os
from typing import Optional, List, Tuple

import torch
import torch_npu
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.layers import TensorEmbedding
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_listdir
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.models.base.config import QuantizationConfig
from .buildmlp.build_mlp import build_vision_tower, build_vision_projector
from .buildmlp.build_mlp_4k import build_vision_tower as build_vision_tower_4k
from .buildmlp.build_mlp_4k import build_vision_projector as build_vision_projector_4k
from .ixc_utils import HD_transform as hd_transform

_INTERPOLATE_FALL_BACK = torch_npu.npu.get_device_name().startswith("Ascend310P")


def format_lora_a_key(base_weight_prefix):
    return f"{base_weight_prefix}.Plora_A.weight"


def format_lora_b_key(base_weight_prefix):
    return f"{base_weight_prefix}.Plora_B.weight"


def format_w8a8sc_lora_a_key(base_weight_prefix):
    lora_a_key = f"{base_weight_prefix}.Plora_A.weight"
    lora_a_key = lora_a_key.replace("language_model.", "")
    return lora_a_key


def format_w8a8sc_lora_b_key(base_weight_prefix):
    lora_b_key = f"{base_weight_prefix}.Plora_B.weight"
    lora_b_key = lora_b_key.replace("language_model.", "")
    return lora_b_key


class MultiModalLLm(FlashForCausalLM):
    def __init__(self, config, weights):
        if getattr(config, "text_config", None):
            if not config.quantize:
                setattr(config.text_config, 'quantize', None)
            else:
                setattr(config.text_config, 'quantize', config.quantize)
            setattr(config.text_config, 'quantization_config', QuantizationConfig({}))
            super().__init__(config.text_config, weights)
        else:
            super().__init__(config, weights)
        self.config = config
        self.weights = weights
        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vit = None
        self.language_model = None
        self.im_mask = None
        self.version = None
        self.lmhead_prefix = "output"
        self.model_prefix = "model"
        try:
            self.version = self._get_version(self.config.max_length)
        except KeyError as e:
            error_message = str(e)
            logger.error("error catched: " + error_message)
            raise e
        self.init_vit()
        self.init_llm()
        self.init_multimodal()
    
    
    def init_vit(self):
        if self.version == "4khd-7b":
            self.vit = build_vision_tower_4k()
        else:
            self.vit = build_vision_tower()
        self.init_module_weight(
            self.vit,
            self.weights,
            prefix="vit",
            prefixskip="vision_tower.vision_model.post_layernorm"
        )

    def init_llm(self):
        model_type = "internlm2"
        current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        supported_models = []
        for foldername in safe_listdir(current_path):
            is_folder = os.path.isdir(os.path.join(current_path, foldername))
            skip_base_folder = foldername != "base"
            skip_invalid_folder = not foldername.startswith("_")
            if is_folder and skip_base_folder and skip_invalid_folder:
                supported_models.append(foldername)

        if model_type not in supported_models:
            msg = (
                f"unsupported model type: {model_type};"
                f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
            )
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(msg)
       
        model_file_dir_name = f"atb_llm.models.{model_type}.v2."
        model_file_name = 'flash_causal'
        module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
        cls = getattr(module, model_cls_name)
        if self.config.quantize == "w8a8sc":
            self.lmhead_prefix = "language_model.lm_head"
            self.model_prefix = "language_model.model"
        self.language_model = cls(self.config,
                                  self.weights,
                                  lmhead_prefix=self.lmhead_prefix,
                                  model_prefix=self.model_prefix)
        self.language_model.skip_word_embedding = True

    @abc.abstractmethod
    def init_multimodal(self):
        pass

    @abc.abstractmethod
    def prepare_prefill_token(self, multimodalinputs, processor):
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
        self.language_model.adapter_manager = self.adapter_manager
        return self.language_model.forward(input_ids, 
                                          position_ids,
                                          is_prefill,
                                          kv_cache,
                                          block_tables,
                                          slots,
                                          input_lengths,
                                          max_seq_len,
                                          lm_head_indices,
                                          im_mask=self.im_mask,
                                          **kwargs)


class FlashInternlmxcomposer2ForCausalLM(MultiModalLLm):
    def __init__(self, config, weights):
        self.vision_proj = None
        self.im_mask = None
        super().__init__(config, weights)
        self.rank = ENV.rank
        self.config = config

        # 由于框架侧generation_config.json优先级高于config.json，会覆盖，当前实现方案需要强制指定max_length
        if self.version == "4khd-7b":
            self.max_length = 16384
        else:
            self.max_length = 4096
        print_log(self.rank, logger.info, f'Set max_length to {self.max_length}')
        self.vocab_size = self.config.vocab_size

        if self.version == "4khd-7b":
            self.plora_glb_gn = nn.Parameter(torch.zeros([1, 1, 4096]))
            self.plora_sub_gn = nn.Parameter(torch.zeros([1, 1, 1, 4096]))
            self.vis_processor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.vis_processor = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size),
                                interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.tok_embeddings = TensorEmbedding(
            prefix=f"{self.model_prefix}.tok_embeddings", weights=weights
        )
        for p in self.tok_embeddings.parameters():
            p.requires_grad = False

    @staticmethod
    def _get_version(max_len):
        if max_len == 1600:
            version = 'vl-7b'
        elif max_len == 4480:
            version = '4khd-7b'
        else:
            msg = "Currently only internlmxc2-vl-7b, internlmxc2-4khd-7b are supported. \
                           If it is the above model, \
                           please check whether the max_length field in generation_config.json is standardized"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(msg)
        return version

    def update_adapter_manager(self):
        self.adapter_manager.base_model = self
        self.adapter_manager.format_lora_a_key = format_lora_a_key
        self.adapter_manager.format_lora_b_key = format_lora_b_key
        if self.config.quantize == "w8a8sc":
            self.adapter_manager.format_lora_a_key = format_w8a8sc_lora_a_key
            self.adapter_manager.format_lora_b_key = format_w8a8sc_lora_b_key
        self.adapter_manager.enable_single_adapter_only = True

    def init_module_weight(self, module, weights, prefix="vision_model", prefixskip=None):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            if prefixskip and prefixskip in model_weight:
                continue
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"{prefix}.{model_weight}"), requires_grad=False
                )
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    def init_multimodal(self):
        if self.version == "4khd-7b":
            self.vision_proj = build_vision_projector_4k()
        else:
            self.vision_proj = build_vision_projector()
        self.init_module_weight(self.vision_proj, self.weights, prefix="vision_proj")

    def prepare_prefill_token(self, multimodalinputs, processor):
        meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n' \
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by ' \
        'Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n' \
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently ' \
        'in the language chosen by the user such as English and 中文.\n' \
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending ' \
        'and articulating responses effectively based on the provided image.'
        history = []
        text = multimodalinputs.text
        image = multimodalinputs.image
        batch_size = multimodalinputs.batch_size
        
        image = self.encode_img(image)
        inputs, self.im_mask = self.interleav_wrap_chat(processor, text, image, history, meta_instruction)

        if self.im_mask is not None:
            self.im_mask = self.im_mask.squeeze(0).unsqueeze(-1).to(torch.float16)
            self.im_mask = torch.cat([self.im_mask for _ in range(batch_size)], dim=0)

        return inputs.get("inputs_embeds", None).squeeze(0)

    def encode_img(self, image_path, hd_num=25):
        if isinstance(image_path, str):
            img = safe_open_image(Image, image_path).convert('RGB')
            if self.version == "4khd-7b":
                img = hd_transform(img, hd_num=hd_num)
            image = self.vis_processor(img).unsqueeze(0).to(self.device)
            img.close()
        else:
            print_log(self.rank, logger.error, 'image type is not supported')
            raise TypeError('image type is not supported')

        img_embeds, atts_img, img_target = self.img2emb(image)
        return img_embeds

    def img2emb(self, image):
        img_embeds = None
        if self.version == "4khd-7b":
            if _INTERPOLATE_FALL_BACK:
                image = image.cpu()
            img_embeds, img_split = self.vit([image], 
            self.plora_glb_gn, self.plora_sub_gn)
            if len(img_split) > 1:
                print_log(self.rank, logger.error, 'Batch Size > 1 is not supported.')
                raise Exception('Batch Size > 1 is not supported.')
        else:
            img_embeds = self.vit(image.to(self.device))
        img_embeds = self.vision_proj(img_embeds)
        atts_img = torch.ones(
            img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        img_target = torch.ones(
            img_embeds.size()[:2], dtype=torch.long).to(
                img_embeds.device) * -100

        return img_embeds, atts_img, img_target

    def interleav_wrap_chat(self, tokenizer, query, image, history, meta_instruction):
        prompt = ''
        if meta_instruction:
            prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        for record in history:
            prompt += f"[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n" \
                       "[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

        im_len = image.shape[1]
        image_nums = len(image)
        parts = prompt.split('<ImageHere>')
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0

        if len(parts) != image_nums + 1:
            raise ValueError('Invalid <ImageHere> prompt format.')
    
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = tokenizer(part, return_tensors='pt').to(self.device)
                part_embeds = self.tok_embeddings(
                    part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_embeds.append(image[idx].unsqueeze(0))
                wrap_im_mask.append(torch.ones(1, image[idx].shape[0]))
                temp_len += im_len
    
            if temp_len > self.max_length:
                break
    
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device).bool()
        inputs = {
            'inputs_embeds': wrap_embeds
        }
        return inputs, wrap_im_mask