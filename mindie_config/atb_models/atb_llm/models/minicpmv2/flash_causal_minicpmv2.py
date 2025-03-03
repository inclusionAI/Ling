# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
# This code is referenced from HuggingFace's MiniCPM-V-2 developed by openbmb.
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
"""PyTorch MiniCPMV2 model."""

import importlib
import os
import json
import math
from copy import deepcopy
from typing import Optional, List, Tuple

import timm
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers.configuration_utils import PretrainedConfig
import torch
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from peft.utils.other import ModulesToSaveWrapper

from atb_llm.utils.log import logger
from atb_llm.utils.file_utils import safe_listdir
from .resampler import Resampler
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.multimodal_utils import safe_open_image


MSG_CONTENT = "content"
INPUT_IDS = "input_ids"


def get_supported_models():
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    supported_models = []
    for foldername in safe_listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)
    return supported_models


def get_llm_model(model_type):
    supported_models = get_supported_models()
    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type};"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )
    model_type = 'minicpm'
    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name = 'flash_causal'
    module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
    module = importlib.import_module(module_path)
    model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class MultiModalLLm(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if not config.quantize:
            setattr(config, 'quantize', None)
        else:
            setattr(config, 'quantize', config.quantize)
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.vocab_size
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
                    weights.get_tensor(f"vpm.{vision_weight}"),
                    requires_grad=False
                )
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    def init_vit(self):
        self.vision_tower = timm.create_model(
            self.config.vision_encoder,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True
        )

        if isinstance(self.vision_tower, timm.models.VisionTransformer):
            if self.vision_tower.attn_pool is not None:
                self.vision_tower.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            self.vision_tower.blocks = self.vision_tower.blocks[:-1]
        
        self.init_visiontowerweight(self.vision_tower, self.weights)

    def init_llm(self):
        self.model_type = 'minicpm'
        model_cls = get_llm_model("minicpm")
        self.language_model = model_cls(self.config,
                                  self.weights,
                                  "llm.lm_head",
                                  "llm.model")
        self.language_model.skip_word_embedding = True
        

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


class FlashMinicpmv2ForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.vision_dim = self.vision_tower.embed_dim
        self.embed_dim = self.config.hidden_size
        self.init_resampler(self.embed_dim, self.vision_dim)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.multi_modal_projector = None
        self.init_multimodal()
        self.transform = self.init_transform()

    @staticmethod
    def init_resamplerweight(module, weights):
        resampler_weights = [resampler_weight for resampler_weight in module.state_dict().keys()]
        for resampler_weight in resampler_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"resampler.{resampler_weight}"),
                    requires_grad=False
                )
            resampler_weight_list = resampler_weight.split(".")
            target_module = module
            for nxt_module in resampler_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, resampler_weight_list[-1], saved_weight)

    def init_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    def init_resampler(self, embed_dim, vision_dim):
        self.resampler = Resampler(
            grid_size=int(math.sqrt(self.config.query_num)),
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

        self.init_resamplerweight(self.resampler, self.weights)

    def init_multimodal(self):
        pass
    
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
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def find_best_resize(self, original_size, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            try:
                r = width / height
                height = int(scale_resolution / math.sqrt(r))
            except ZeroDivisionError as e:
                logger.error('r or height divided by zero!')
                raise ZeroDivisionError from e
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return (best_width, best_height)

    def vpm_forward_features(self, pixel_value):
        if isinstance(self.vision_tower, ModulesToSaveWrapper):
            if self.vision_tower.disable_adapters or (self.vpm.active_adapter not in self.vpm.modules_to_save):
                return self.vision_tower.original_module.forward_features(pixel_value)
            return self.vision_tower.modules_to_save[self.vpm.active_adapter].forward_features(pixel_value)
        else:    
            return self.vision_tower.forward_features(pixel_value)

    def get_vision_embedding(self, pixel_values):
        res = []
        dtype = self.config.torch_dtype

        def process_each_pixel(pixel_value, dtype, config, vpm, resampler):
            height, width = pixel_value.shape[-2:]
            try:
                patch_height = height / config.patch_size
                patch_width = width / config.patch_size
            except ZeroDivisionError as e:
                logger.error('patch_height or patch_width divided by zero!')
                raise ZeroDivisionError from e
            target_size = (math.ceil(patch_height), math.ceil(patch_width))
            vision_embedding = self.vpm_forward_features(pixel_value.unsqueeze(0).type(dtype))
            
            if hasattr(vpm, 'num_prefix_tokens') and vpm.num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, vpm.num_prefix_tokens:]
            return resampler(vision_embedding, target_size)

        for pixel_value in pixel_values:
            result = process_each_pixel(pixel_value, dtype, self.config, self.vision_tower, self.resampler)
            res.append(result)
        return torch.vstack(res)

    def ensure_divide(self, length, patch_size):
        try:
            patch_length = length / patch_size
        except ZeroDivisionError as e:
            logger.error('patch_length divided by zero!')
            raise ZeroDivisionError from e
        return max(round(patch_length) * patch_size, patch_size)

    def get_refine_size(
        self, original_size, grid, scale_resolution, patch_size, allow_upscale=False
    ):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)
        try:
            grid_width = refine_width / grid_x
            grid_height = refine_height / grid_y
        except ZeroDivisionError as e:
            logger.error('grid_width or grid_height divided by zero!')
            raise ZeroDivisionError from e

        best_grid_size = self.find_best_resize(
            (grid_width, grid_height),
            scale_resolution,
            patch_size,
            allow_upscale=allow_upscale,
        )

        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

        return refine_size

    def split_to_patches(self, image, grid):
        patches = []
        width, height = image.size
        try:
            grid_x = int(width / grid[0])
            grid_y = int(height / grid[1])
        except ZeroDivisionError as e:
            logger.error('grid_x or grid_y divided by zero!')
            raise ZeroDivisionError from e

        for i in range(0, height, grid_y):
            images = []
            for j in range(0, width, grid_x):
                box = (j, i, j + grid_x, i + grid_y)
                patch = image.crop(box)
                images.append(patch)
            patches.append(images)

        return patches

    def slice_image(
        self, image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
    ):
        original_size = image.size
        original_width, original_height = original_size
        try:
            log_ratio = math.log(original_width / original_height)
            ratio = original_width * original_height / (scale_resolution * scale_resolution)
        except ZeroDivisionError as e:
            logger.error('log_ratio or ratio divided by zero!')
            raise ZeroDivisionError from e
        multiple = min(math.ceil(ratio), max_slice_nums)

        source_image = None
        best_grid = None
        patches = []

        if multiple <= 1 or never_split:
            # dont need to slice, upsample
            best_size = self.find_best_resize(
                original_size, scale_resolution, patch_size, allow_upscale=True
            )
            source_image = image.resize(best_size, Image.Resampling.BICUBIC)
        else:
            candidate_split_grids_nums = []
            for i in [multiple - 1, multiple, multiple + 1]:
                if i == 1 or i > max_slice_nums:
                    continue
                candidate_split_grids_nums.append(i)

            # source image, down-sampling and ensure divided by patch_size
            best_resize = self.find_best_resize(original_size, scale_resolution, patch_size)
            source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
            candidate_grids = []

            # find best grid
            for split_grids_nums in candidate_split_grids_nums:
                m = 1
                while m <= split_grids_nums:
                    if split_grids_nums % m == 0:
                        try:
                            splited_grids_nums = split_grids_nums // m
                        except ZeroDivisionError as e:
                            logger.error('splited_grids_nums divided by zero!')
                            raise ZeroDivisionError from e
                        candidate_grids.append([m, splited_grids_nums])
                    m += 1

            best_grid = [1, 1]
            min_error = float("inf")
            for grid in candidate_grids:
                try:
                    divided_grid = grid[0] / grid[1]
                except ZeroDivisionError as e:
                    logger.error('divided_grid divided by zero!')
                    raise ZeroDivisionError from e
                error = abs(log_ratio - math.log(divided_grid))
                if error < min_error:
                    best_grid = grid
                    min_error = error

            refine_size = self.get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )

            refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
            patches = self.split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                elif self.training:
                    dtype = self.config.torch_dtype
                    device = self.device
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224), device=device, dtype=dtype
                    )
                    vision_hidden_states.append(self.get_vision_embedding(dummy_image))
                else:
                    vision_hidden_states.append([])
        else:
            vision_hidden_states = data["vision_hidden_states"]
        vllm_embedding = (
            self.language_model.model.embed_tokens(data["input_ids"]) * self.language_model.config.scale_emb
        )
        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [
                            torch.arange(r[0], r[1], dtype=torch.long)
                            for r in cur_image_bound
                        ]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def pad(self, orig_items, key, max_length=None, padding_value=0, padding_side=None): 
        items = []
        if isinstance(orig_items[0][key], list):
            if not isinstance(orig_items[0][key][0], torch.Tensor):
                raise AssertionError()
            for it in orig_items:
                for tr in it[key]:
                    items.append({key: tr})
        else:
            if not isinstance(orig_items[0][key], torch.Tensor):
                raise AssertionError()
            items = orig_items

        batch_size = len(items)
        shape = items[0][key].shape
        dim = len(shape)
        if dim > 3:
            raise AssertionError()
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item[key].shape[-1] for item in items))
        min_length = min(item[key].shape[-1] for item in items)
        dtype = items[0][key].dtype

        if dim == 1:
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 2:
            if max_length == min_length:
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = (
                torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
                + padding_value
            )

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()

        return tensor

    def get_grid_placeholder(self, tokenizer, grid, query_num):
        image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
        )

        cols = grid[0]
        rows = grid[1]
        slices = []
        for _ in range(rows):
            lines = []
            for _ in range(cols):
                lines.append(image_placeholder)
            slices.append("".join(lines))
        slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
        return slice_placeholder

    def get_slice_image_placeholder(self, image, tokenizer):
        image_placeholder = (
            tokenizer.im_start
            + tokenizer.unk_token * self.config.query_num
            + tokenizer.im_end
        )

        slice_images = []

        source_image, patches, best_grid = self.slice_image(
            image,
            self.config.max_slice_nums,
            self.config.scale_resolution,
            self.config.patch_size,
        )

        slice_images.append(source_image)
        final_placeholder = image_placeholder

        patcheslen = len(patches)
        if patcheslen > 0:
            patcheslen0 = len(patches[0])
            for i in range(patcheslen):
                for j in range(patcheslen0):
                    slice_images.append(patches[i][j])

            final_placeholder += self.get_grid_placeholder(
                tokenizer, best_grid, self.config.query_num
            )

        return slice_images, final_placeholder

    def generate_vllm(
        self,
        data_list=None,
        img_list=None,
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
    ):

        if not data_list:
            raise AssertionError()
        bs = len(data_list)
        if not img_list:
            img_list = [[] for i in range(bs)]
        if bs != len(img_list):
            raise AssertionError()

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(self.transform(img).to(self.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                res,
                _,
            ) = self.get_vllm_embedding(model_inputs)
        return res

    def prepare_prefill_token(self, multimodalinputs, processor, **kwargs):
        tokenizer = processor

        image = multimodalinputs.image
        image = safe_open_image(Image, image)
        
        text = multimodalinputs.text
        text = [text]
        if isinstance(text, str):
            text = json.loads(text)
        msgs = deepcopy(text)

        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # msgs to prompt
        prompt = ""
        for i, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            if role not in ["user", "assistant"]:
                raise AssertionError()
            if i == 0:
                if role != "user":
                    raise AssertionError()
                if self.config.slice_mode:
                    images, final_placeholder = self.get_slice_image_placeholder(
                        image, tokenizer
                    )
                    content = final_placeholder + "\n" + content
                else:
                    images = [image]
                    content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * self.config.query_num
                        + tokenizer.im_end
                        + "\n"
                        + content
                    )
            prompt += "<用户>" if role == "user" else "<AI>"
            prompt += content
        prompt += "<AI>"
        final_input = prompt
        with torch.inference_mode():
            res = self.generate_vllm(
                data_list=[final_input],
                max_inp_length=2048,
                img_list=[images],
                tokenizer=tokenizer,
                )
        image.close()
        return res[0]

    def init_ascend_operations(self, config: PretrainedConfig):
        pass

    def init_ascend_weight(self):
        pass
    
    def _process_list(
        self, tokenizer, data_list: List[str], max_inp_length: Optional[int] = None
    ):
        pad_keys = ["input_ids"]
        input_tensors = []
        for data in data_list:
            input_tensors.append(
                self._convert_to_tensors(tokenizer, data, max_inp_length)
            )
        padded = {}
        for key in pad_keys:
            padded[key] = self.pad(input_tensors, key, padding_side="left").to(self.device)
        padded["image_bound"] = [i["image_bound"] for i in input_tensors]
        return padded

    def _convert_to_tensors(
        self, tokenizer, input_str, max_inp_length: Optional[int] = None
    ):
        if tokenizer.add_bos_token:
            input_ids = tokenizer.encode(input_str)
        else:
            input_ids = [tokenizer.bos_id] + tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0]
        # 跳过 im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bound = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )

        model_input = {}
        model_input["input_ids"] = input_ids.unsqueeze(0).to(self.device)
        model_input["image_bound"] = image_bound

        return model_input

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, image_token_id):
        mask = (input_ids == image_token_id)
        inputs_embeds[mask] = image_features
        return inputs_embeds
