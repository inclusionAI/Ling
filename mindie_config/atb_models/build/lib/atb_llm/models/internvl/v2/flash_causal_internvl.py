# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from typing import Optional, List, Tuple
import importlib

import torch
from torch import nn
import numpy as np

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.log.logging import logger, print_log
from atb_llm.models.internvl.modeling_intern_vit import InternVisionModel
from atb_llm.models.internvl.data_preprocess_internvl import (
    load_video, load_and_preprocess_image, create_standardization_params,
    internvl_tensor_parallel_split, IMAGENET_MEAN, IMAGENET_STD
)
from atb_llm.models.internvl.input_builder_internvl import INTERNVL_SYSTEM_PROMPTS
from atb_llm.utils.dist import initialize_torch_distributed
from atb_llm.utils.layers.linear.linear import ColumnLinear, RowLinear
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.utils import shm_utils

# Preprocessing params
RESCALE_FACTOR = 1 / 255
CONV_CHANNELS = 3
CONV_GROUPS = 3
IMAGE_SIZE = 448
MAX_NUM_PATCHES = 12

EOS_TOKEN_ID = 92542


class FlashInternvlForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.config.eos_token_id = EOS_TOKEN_ID
        self.weights = weights # id相等，引用传递
        self.dtype = weights.dtype
        self.vision_config = config.vision_config
        self.llm_config = config.llm_config
        self.llm_config.quantize = None
        if self.dtype != torch.float16:
            logger.error(
                f"Unsupported dtype: {self.dtype}, supported dtypes are `float16`."
                "Please check the value of 'torch_dtype' in config.json",
                ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID
            )
            raise ValueError(
                f"Unsupported dtype: {self.dtype}, supported dtypes are `float16`."
                "Please check the value of 'torch_dtype' in config.json"
            )
        # 图片处理相关参数
        self.downsample_ratio = config.downsample_ratio
        self.vit_hidden_size = self.vision_config.hidden_size
        self.llm_hidden_size = self.llm_config.hidden_size
        self.image_size = config.force_image_size or self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size
        self.select_layer = config.select_layer
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.neftune_alpha = None
        self.im_mask = None
        self.template = config.template
        self.ps_version = config.ps_version
        if self.template not in ['Hermes-2', 'internlm2-chat', 'phi3-chat', 'internvl2_5']:
            raise ValueError(
                f"Unsupported template {self.template}, supported templates are `Hermes-2`, "
                "`internlm2-chat`, `phi3-chat`, `internvl2_5`. Please check the value of 'template' in config.json"
            )
        if self.ps_version not in ['v1', 'v2']:
            raise ValueError(
                f"Unsupported ps_version {self.ps_version}, supported templates are `v1` and `v2`."
                "Please check the value of 'ps_version' in config.json"
            )

        self.npu_id = weights.device.index
        self.process_group, self.device = initialize_torch_distributed(self.tp_rank, self.npu_id, self.tp_world_size)
        self.init_llm_model_type()
        self.init_vision_model()
        self.init_mlp_projector()
        self.init_language_model()
        self.init_normalizer()
        self.llm_embedding_layer = self.language_model.get_embedding_layer()
        self.processor = safe_get_tokenizer_from_pretrained(config.model_name_or_path, 
                                                            trust_remote_code=self.trust_remote_code)
        self.img_begin_id = self.processor.encode("<img>")[-1]
        self.img_end_id = self.processor.encode("</img>")[-1]
        self.img_context_token_id = self.processor.encode("<IMG_CONTEXT>")[-1]

    def init_module_weight(self, module, weights, prefix="model", prefixskip=None):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            if prefixskip and prefixskip in model_weight:
                continue
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"{prefix}.{model_weight}"), requires_grad=False
                )
            saved_weight = internvl_tensor_parallel_split(model_weight, prefix, \
                self.tp_rank, self.tp_world_size, saved_weight)
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    def init_llm_model_type(self):
        llm_model_architectures = self.llm_config.architectures[0]
        llm_model_num_hidden_layers = self.llm_config.num_hidden_layers
        if llm_model_architectures == 'Qwen2ForCausalLM' and llm_model_num_hidden_layers == 24:
            self.llm_model_type = "qwen2" # qwen: VL2-1B
        elif llm_model_architectures == 'InternLM2ForCausalLM':
            self.llm_model_type = "internlm2" # internlm: VL2-2B、VL2-8B、VL2-20B
        elif llm_model_architectures == 'LlamaForCausalLM':
            self.llm_model_type = "llama" # llama, yi: VL2-40B、VL2-76B
        else:
            logger.error("Currently only InternVL‑Chat‑V1‑2、InternVL‑Chat‑V1‑5、InternVL2 are supported. "
                         "Please Check `config.json`.",
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise KeyError("Currently only InternVL‑Chat‑V1‑2、InternVL‑Chat‑V1‑5、InternVL2 are supported. "
                           "Please Check `config.json`.")

    def init_normalizer(self):
        weight, bias = create_standardization_params(IMAGENET_MEAN, IMAGENET_STD, RESCALE_FACTOR, CONV_CHANNELS)
        self.normalizer = nn.Conv2d(in_channels=CONV_CHANNELS, out_channels=CONV_CHANNELS, kernel_size=1, \
            groups=CONV_GROUPS)
        self.normalizer.weight = nn.Parameter(data=weight, requires_grad=False)
        self.normalizer.bias = nn.Parameter(data=bias, requires_grad=False)
        self.normalizer.npu()
        # Normalizer warmup
        self.normalizer(torch.randn(MAX_NUM_PATCHES, CONV_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device='npu'))

    def init_vision_model(self):
        self.vision_model = InternVisionModel(self.vision_config, self.process_group).to(dtype=self.dtype)
        self.init_module_weight(self.vision_model, self.weights, prefix="vision_model")
        self.vision_model = self.vision_model.to(self.device)

    def init_mlp_projector(self):
        if self.downsample_ratio == 0:
            raise ZeroDivisionError("Downsample ratio will be zero")
        input_dim = self.vit_hidden_size * int(np.divide(1, self.downsample_ratio)) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            ColumnLinear(input_dim, self.llm_hidden_size, gather_output=False, process_group=self.process_group),
            nn.GELU(),
            RowLinear(self.llm_hidden_size, self.llm_hidden_size, process_group=self.process_group)
        ).to(dtype=self.dtype)
        self.init_module_weight(self.mlp1, self.weights, prefix="mlp1")
        self.mlp1 = self.mlp1.to(self.device)

    def init_language_model(self):
        model_type = self.llm_model_type # internlm2
        module_path = f"atb_llm.models.{model_type}.v2.flash_causal_{model_type}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error(f'Unsupported module: {module_path}.',
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(f'Unsupported module: {module_path}.') from e
        model_cls_name = f"Flash{model_type.capitalize()}ForCausalLM"
        cls = getattr(module, model_cls_name)
        self.language_model = cls(self.config,
                                  self.weights,
                                  lmhead_prefix="language_model.output",
                                  model_prefix="language_model.model")
        self.language_model.skip_word_embedding = True


    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        if scale_factor == 0:
            raise ZeroDivisionError("Scale factor will be zero")
        x = x.view(n, w, int(h * scale_factor), int(np.divide(c, scale_factor)))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        if scale_factor == 0:
            raise ZeroDivisionError("Scale factor will be zero")
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(np.divide(c, scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            print_log(self.tp_rank, logger.warnings, 
                      "In ps_version 'v1', the height and width have not been swapped back, "
                      "which results in a transposed image.")
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def noised_embed(self, vit_embeds, noise_alpha=5):
        dims = torch.tensor(vit_embeds.size(1) * vit_embeds.size(2))
        if dims == 0:
            raise ZeroDivisionError("Dim of the tensor is zero")
        mag_norm = np.divide(noise_alpha, torch.sqrt(dims))
        noise = torch.zeros_like(vit_embeds).uniform_(-mag_norm, mag_norm)
        return vit_embeds + noise

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def prepare_prefill_token_service(self, input_ids):
        input_embeds = self.llm_embedding_layer(input_ids)
        sequence_length, embedding_size = input_embeds.shape
        input_ids = input_ids.reshape(sequence_length)
        if torch.any(torch.eq(input_ids, self.img_begin_id)):
            img_bos_set = torch.where(torch.eq(input_ids, self.img_begin_id))[0].detach().cpu().tolist()
            img_eos_set = torch.where(torch.eq(input_ids, self.img_end_id))[0].detach().cpu().tolist()
            batch_images = []
            batch_size_list = []

            for img_bos, img_eos in zip(img_bos_set, img_eos_set):
                if img_eos - img_bos < 2:
                    continue
                image_pixel_value = shm_utils.get_data_from_shm(input_ids[img_bos + 1], input_ids[img_bos + 2], 
                    dtype=np.uint8, device=self.device).to(self.dtype)
                
                batch_images.append(image_pixel_value)
                batch_size_list.append(image_pixel_value.size(0))

            batch_images = torch.cat(batch_images, dim=0)
            batch_images = self.normalizer(batch_images.float()).half()

            vit_embeds = self.extract_feature(batch_images)
            vit_embeds = vit_embeds.to(self.dtype).to(self.device)
            
            pre_index = 0
            for img_bos, img_eos, batch_size in zip(img_bos_set, img_eos_set, batch_size_list):
                single_vit_embeds = vit_embeds[pre_index : pre_index + batch_size].reshape(-1, embedding_size)
                pre_index += batch_size
                try:
                    input_embeds[img_bos + 1 : img_eos] = single_vit_embeds
                except Exception as e:
                    error_msg = f'{e} \ninput_embeds[selected].shape={input_embeds[img_bos + 1 : img_eos].shape}, '\
                            f'vit_embeds.shape={single_vit_embeds.shape}\n'\
                            f'Please check whether shape of input_embeds[selected] matches the shape of vit_embeds.\n'\
                            f'If not, please check whether self.img_context_token_id '\
                            f'and the token-id of "<IMG_CONTEXT>" in processor are the same'
                    logger.error(error_msg,
                                 ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                    raise ValueError(error_msg) from e

        input_embeds = input_embeds.reshape(-1, embedding_size)
        return input_embeds
    
    def prepare_prefill_token(self, multimodalinputs, processor):
        text = multimodalinputs.text
        image = multimodalinputs.image
        video = multimodalinputs.video
        current_query = ""
        if image is not None:
            use_dynamic_prepro = False if self.ps_version == "v1" else True
            pixel_values = load_and_preprocess_image(image, normalizer=self.normalizer, \
                use_dynamic_prepro=use_dynamic_prepro).to(self.dtype).to(self.device)
            vit_embeds = self.extract_feature(pixel_values).to(self.dtype).to(self.device)
            image_tokens_num = self.num_image_token * vit_embeds.shape[0]
            current_query = (f'<img>{"<IMG_CONTEXT>" * image_tokens_num}</img>\n')
        elif video is not None:
            pixel_values, num_path_list = load_video(video)
            pixel_values = pixel_values.to(self.dtype).to(self.device)
            vit_embeds = self.extract_feature(pixel_values).to(self.dtype).to(self.device)
            current_query = ""
            for i, num_patch in enumerate(num_path_list):
                current_query += (f'Frame{i+1}: '
                    f'<img>{"<IMG_CONTEXT>" * num_patch * self.num_image_token}</img>\n')

        system_prompt = INTERNVL_SYSTEM_PROMPTS[self.ps_version][self.template]
        texts = ('<|im_start|>system\n'
                f'{system_prompt}<|im_end|><|im_start|>user\n')
        texts += current_query
        texts += (f'{text}<|im_end|><|im_start|>assistant\n')

        input_ids = processor.encode(texts)
        input_ids = torch.tensor(input_ids, requires_grad=False).to(self.device)
        input_embeds = self.llm_embedding_layer(input_ids)
        sequence_length, embedding_size = input_embeds.shape

        input_ids = input_ids.reshape(sequence_length)
        vit_embeds = vit_embeds.reshape(-1, embedding_size)
        selected = (input_ids == self.img_context_token_id)
        
        try:
            input_embeds[selected] = input_embeds[selected] * torch.zeros(1, dtype=self.dtype,
                                                        device=self.device) + vit_embeds.reshape(-1, embedding_size)
        except Exception as e:
            error_msg = f'{e} \ninput_embeds[selected].shape={input_embeds[selected].shape}, '\
                        f'vit_embeds.shape={vit_embeds.shape}\n'\
                        f'Please check whether shape of input_embeds[selected] matches the shape of vit_embeds.\n'\
                        f'If not, please check whether self.img_context_token_id '\
                        f'and the token-id of "<IMG_CONTEXT>" in processor are the same'
            logger.error(error_msg,
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg) from e

        input_embeds = input_embeds.reshape(-1, embedding_size)
        return input_embeds

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
                                          im_mask=self.im_mask,
                                          **kwargs)
