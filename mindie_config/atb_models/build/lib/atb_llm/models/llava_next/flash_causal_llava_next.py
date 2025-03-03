# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
import os
import math

import av
import torch
import numpy as np 
from PIL import Image
from torch import nn
from transformers import AutoProcessor

from atb_llm.utils.shm_utils import decode_shape_from_int64, get_data_from_shm
from ..llava.flash_causal_llava import FlashLlavaForCausalLM, LlavaConfig
from .data_preprocess_llava_next import get_anyres_image_grid_shape, unpad_image, \
    image_size_to_num_patches, read_video_pyav
from ..base.model_utils import safe_from_pretrained


PYTORCH_TENSOR = "pt"


class LlavaNextConfig(LlavaConfig):

    model_type = "llava_next"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32001,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_grid_pinpoints=None,
        tie_word_embeddings=False,
        video_token_index=None,
        spatial_pool_mode="average",
        spatial_pool_stride=2,
        **kwargs,
    ):
        super().__init__(vision_config,
                         text_config,
                         ignore_index,
                         image_token_index,
                         projector_hidden_act,
                         vision_feature_select_strategy,
                         vision_feature_layer,
                         **kwargs)
        self.video_token_index = video_token_index
        self.spatial_pool_mode = spatial_pool_mode
        self.spatial_pool_stride = spatial_pool_stride
        self.max_position_embeddings = 4096
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.image_grid_pinpoints = image_grid_pinpoints
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_model_name = self.text_config["_name_or_path"]


class LlavaNextVideoPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        mode = config.spatial_pool_mode
        stride = config.spatial_pool_stride
        out_channels = getattr(config, "spatial_pool_out_channels", config.vision_config.hidden_size)
        self.image_size = config.vision_config.image_size // config.vision_config.patch_size**2

        if mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        elif mode == "conv":
            self.pool = nn.Conv2d(
                in_channels=config.vision_config.hidden_size,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
            )
        else:
            raise ValueError(f"Unknown pooling mode: {mode}. Has to be one of [`average`, `max`, `conv`]")

    def forward(self, image_features):
        ori_width = int(math.sqrt(image_features.shape[1] * self.image_size // self.image_size))
        ori_height = int(ori_width * self.image_size // self.image_size)

        batch_size, _, dim = image_features.shape
        image_features_spatial = image_features.view(batch_size, ori_height, ori_height, dim).permute(0, 3, 1, 2)
        image_features_spatial_pool = self.pool(image_features_spatial)

        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()


class FlashLlava_nextForCausalLM(FlashLlavaForCausalLM):
    def __init__(self,
                 config,
                 weights):
        super().__init__(config, weights)
        self.config = config
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.text_config.pad_token_id \
            if self.config.text_config.pad_token_id is not None else -1
        self.video_token_id = self.config.video_token_index if self.config.video_token_index is not None else -2
        self.vision_resampler = None
        
        self.vision_feature_layer = self.config.vision_feature_layer
        self.vision_feature_select_strategy = self.config.vision_feature_select_strategy
        self.architectures = self.config.architectures

        num_hidden_layers = config.text_config.num_hidden_layers
        if num_hidden_layers == 60 and self.architectures[0] == "LlavaNextForConditionalGeneration":
            self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path, use_fast=False)
        else:
            self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path)
        self.init_video()

    def init_video(self):
        if "LlavaNextVideoForConditionalGeneration" in self.architectures:
            self.vision_resampler = LlavaNextVideoPooler(self.config)

    def prepare_prefill_token_service(self, input_ids):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_bos = torch.where(torch.eq(input_ids, self.image_token_id))[0]
        video_bos = torch.where(torch.eq(input_ids, self.video_token_id))[0]

        if image_bos.shape[0] == 0 and video_bos.shape[0] == 0:
            return inputs_embeds

        image_num = image_bos.shape[0] // 2
        video_num = video_bos.shape[0] // 2
        if image_num != 0:
            batch_images = []
            batch_image_sizes = []
            for i in range(0, image_num * 2, 2):
                image_pixel_values = get_data_from_shm(input_ids[image_bos[i] + 1], 
                    input_ids[image_bos[i] + 2], dtype=np.float16, device=self.device)
                if image_pixel_values is None:
                    continue
                image_size = decode_shape_from_int64(input_ids[image_bos[i] + 3])[1:]
                image_size = torch.tensor([image_size])
                batch_images.append(image_pixel_values)
                batch_image_sizes.append(image_size)
            batch_images = torch.cat(batch_images, dim=1).squeeze(0)
            batch_image_sizes = torch.cat(batch_image_sizes, dim=0)
            image_features = self._get_image_features(batch_images, batch_image_sizes)
            image_features, feature_lens = self.pack_image_features(image_features, 
                                                                    batch_image_sizes, 
                                                                    self.image_newline)
            feature_begin = 0
            feature_end = 0
            lens_idx = 0
            for i in range(0, image_num * 2, 2):
                feature_end += feature_lens[lens_idx]
                inputs_embeds[image_bos[i]: image_bos[i + 1] + 1] = image_features[feature_begin:feature_end]
                feature_begin += feature_lens[lens_idx]
                lens_idx += 1

        for i in range(0, video_num * 2, 2):
            video_pixel_values = get_data_from_shm(input_ids[video_bos[i] + 1],
                input_ids[video_bos[i] + 2], dtype=np.float16, device=self.device)
            video_features = self._get_video_features(video_pixel_values)
            video_features = video_features.flatten(0, 1)
            inputs_embeds[video_bos[i]: video_bos[i + 1] + 1] = video_features
        return inputs_embeds
    
    def prepare_prefill_token(self, multimodalinputs, processor):
        text = multimodalinputs.text
        image = multimodalinputs.image
        video = multimodalinputs.video
        if hasattr(multimodalinputs, 'frames'):
            frames = multimodalinputs.frames
        else:
            frames = 8
        inputs_embeds = None
        if image:
            image = Image.open(image)
            inputs = processor.image_processor(images=image, return_tensors=PYTORCH_TENSOR)
            image.close()
            pixel_values = inputs["pixel_values"].half().npu()
            image_sizes = inputs["image_sizes"]
            if pixel_values is not None and pixel_values.size(0) > 0:
                image_features = self._get_image_features(pixel_values, image_sizes)
                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline)
                image_size = image_features.shape[0]
                image_token_str = processor.tokenizer.decode(self.config.image_token_index)
                replacement_str = "".join([image_token_str] * image_size)
                new_prompt = text.replace(image_token_str, replacement_str, 1)
                input_ids = processor.tokenizer(new_prompt, return_tensors=PYTORCH_TENSOR)["input_ids"].npu()
                inputs_embeds = self.get_input_embeddings()(input_ids)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds = (
                    self._merge_input_ids_with_image_features(
                        image_features,
                        inputs_embeds,
                        input_ids,
                        self.config.image_token_index,
                    )
                )
        if video:
            container = av.open(video)
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / frames).astype(int)
            clip = read_video_pyav(container, indices)
            container.close()
            pixel_values = processor.video_processor(images=clip, return_tensors=PYTORCH_TENSOR)["pixel_values_videos"]
            pixel_values_videos = pixel_values.half().npu()
            if pixel_values_videos is not None and pixel_values_videos.size(0) > 0:
                video_features = self._get_video_features(pixel_values_videos)
                video_features = video_features.flatten(0, 1)
                feature_lens = video_features.size(0)
                image_token_str = processor.tokenizer.decode(self.config.video_token_index)
                replacement_str = "".join([image_token_str] * feature_lens)
                new_prompt = text.replace(image_token_str, replacement_str, 1)
                input_ids = processor.tokenizer(new_prompt, return_tensors=PYTORCH_TENSOR)["input_ids"].npu()
                inputs_embeds = self.get_input_embeddings()(input_ids)
                inputs_embeds = (
                    self._merge_input_ids_with_image_features(
                        video_features,
                        inputs_embeds,
                        input_ids,
                        self.config.video_token_index,
                        )
                    )
        inputs_embeds = inputs_embeds.view(inputs_embeds.shape[0] * inputs_embeds.shape[1],
                                            inputs_embeds.shape[2])
        return inputs_embeds

    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a 
        single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, 
            each of shape `(num_patches, image_length, embed_dim)`)
            List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    def _merge_input_ids_with_image_features_service(self, image_features, inputs_embeds, bos_pos):
        inputs_embeds[bos_pos[0]: bos_pos[1] + 1] = image_features
        return inputs_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, image_token_id):
        mask = (input_ids == image_token_id)
        inputs_embeds[mask] = image_features
        return inputs_embeds
    
    def _get_image_features(self, pixel_values, image_sizes):
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        return image_features

    def _get_video_features(self, pixel_values):
        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * frames, channels, height, width)
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        # Same as image features except that video has pooling layer
        image_features = self.vision_resampler(selected_image_feature)
        image_features = self.multi_modal_projector(image_features)
        return image_features