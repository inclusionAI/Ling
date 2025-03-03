# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""PyTorch Qwen2_vl model."""

from typing import Optional, List, Tuple

import numpy as np
import torch
from atb_llm.utils.dist import initialize_torch_distributed
from atb_llm.utils.shm_utils import get_data_from_shm, decode_shape_from_int64

from ..base.flash_causal_multimodal import MultiModalLLm
from ..qwen2_vl.flash_causal_qwen2_using_mrope import FlashQwen2UsingMROPEForCausalLM
from ..qwen2_vl.modeling_qwen2_vl_vit import Qwen2VisionTransformerPretrainedModel
from ..qwen2_vl.modeling_qwen2_vl_vit_atb import Qwen2VisionTransformerPretrainedModelATB

SPATIAL_MERGE_SIZE = 2
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
PYTORCH_TENSOR = "pt"
MROPE_SECTION = [16, 24, 24]
WEIGHT_KEYS_MAPPING = {
    'attn.qkv.weight': 0,
    'attn.proj.weight': 0,
    'mlp.fc1.weight': 0,
    'mlp.fc2.weight': 1,
}
BIAS_KEYS = ['attn.qkv.bias', 'attn.proj.bias', 'mlp.fc1.bias']
SHM_VALUE_TOKEN_OFFSET = 1
SHAPE_VALUE_TOKEN_OFFSET = 2
IMAGE_THW_TOKEN_OFFSET = 3
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SCALE = 1 / 255
NORMALIZATION_CHANNELS = 3
NORMALIZATION_KERNEL_SIZE = 1
NORMALIZATION_OUTPUT_SIZE = 2 * 14 * 14 * 3
PATCH_SZIE = 2 * 14 * 14


class FlashQwen2vlForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        self.npu_id = weights.device.index
        self.tp_rank = weights.process_group.rank()
        self.tp_world_size = weights.process_group.size()
        self.process_group, self.device = initialize_torch_distributed(self.tp_rank, self.npu_id, self.tp_world_size)
        self.config = config
        self.image_token_id = getattr(self.config, "image_token_id", _IMAGE_TOKEN_ID)
        self.video_token_id = getattr(self.config, "video_token_id", _VIDEO_TOKEN_ID)
        self.vision_start_token_id = getattr(self.config, "vision_start_token_id", VISION_START_TOKEN_ID)
        self.vision_end_token_id = getattr(self.config, "vision_end_token_id", VISION_END_TOKEN_ID)
        self.spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", SPATIAL_MERGE_SIZE)
        self.mrope_section = self.config.mrope_section.get('mrope_section', MROPE_SECTION)
        self.language_model = None
        self.vision_tower = None
        enable_atb_vit = kwargs.pop("enable_atb_vit", True)
        self.enable_atb_vit = enable_atb_vit
        super().__init__(config, weights, **kwargs)
        weight, bias = self.create_standardization_layer(IMAGE_MEAN, IMAGE_STD, IMAGE_SCALE, \
            NORMALIZATION_CHANNELS)
        self.normalizer = torch.nn.Conv1d(in_channels=NORMALIZATION_CHANNELS, out_channels=NORMALIZATION_CHANNELS, \
            kernel_size=NORMALIZATION_KERNEL_SIZE, groups=NORMALIZATION_CHANNELS)
        self.normalizer.weight = torch.nn.Parameter(data=weight, requires_grad=False)  
        self.normalizer.bias = torch.nn.Parameter(data=bias, requires_grad=False)
        self.normalizer.eval()
        self.normalizer.npu()

    def create_standardization_layer(self, mean, std, scale, input_channel):
        if isinstance(std, (list, tuple)):
            std = torch.Tensor(std)
        if isinstance(mean, (list, tuple)):
            mean = torch.Tensor(mean)
        weight = (scale / std).view(input_channel, 1, 1) 
        bias = -mean.view(input_channel) / std.view(input_channel)
        return weight, bias

    def get_input_embeddings(self):
        return self.language_model.transformer.wte

    def qwen2_vl_tensor_parallel_split(
            self,
            key: str,
            tp_rank: int,
            tp_size: int,
            saved_weight: torch.nn.Parameter
    ):
        def split(tensor: torch.Tensor, tp_size: int, tp_rank: int, dim=0):
            if tp_size == 1:
                return tensor
            if not (len(tensor.shape) > 1 or dim == 0):
                raise ValueError("Invalid dimension for splitting. Expected len(tensor.shape) > 1 or dim == 0.")
            if isinstance(tensor, np.ndarray):
                return np.ascontiguousarray(np.split(tensor, tp_size, axis=dim)[tp_rank].copy())
            if tensor.shape[dim] % tp_size != 0:
                raise ValueError(f"Unable to split: shape={tensor.shape} (dim={dim}) tp_size={tp_size}.")
            split_size = tensor.shape[dim] // tp_size
            return tensor.split(split_size, dim=dim)[tp_rank].clone().detach()

        for k, dim in WEIGHT_KEYS_MAPPING.items():
            if k in key:
                saved_weight.data = split(saved_weight.data, tp_size, tp_rank, dim=dim)
                return saved_weight

        if any(k in key for k in BIAS_KEYS):
            saved_weight.data = torch.chunk(saved_weight.data, tp_size)[tp_rank]

        return saved_weight

    def init_module_weight_parallel(self, module, weights):
        vision_weights = [vision_weight for vision_weight in module.state_dict().keys()]
        for vision_weight in vision_weights:
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"visual.{vision_weight}"),
                requires_grad=False
            )
            saved_weight = self.qwen2_vl_tensor_parallel_split(vision_weight, \
                                                               self.tp_rank, self.tp_world_size, saved_weight)
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    def init_vit(self):
        if not self.enable_atb_vit:
            self.vision_tower = Qwen2VisionTransformerPretrainedModel(
                self.config.vision_config, self.process_group
            )
            self.init_module_weight_parallel(self.vision_tower, self.weights)
            self.vision_tower = self.vision_tower.to(self.device)
        else:
            self.vision_tower = Qwen2VisionTransformerPretrainedModelATB(
                self.config.vision_config, self.weights, self.config.max_position_embeddings
            )
            vision_weights = []
            for vision_weight in self.vision_tower.state_dict().keys():
                if vision_weight.startswith("merger") or vision_weight.startswith("patch_embed"):
                    vision_weights.append(vision_weight)
            for vision_weight in vision_weights:
                saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor(f"visual.{vision_weight}"),
                    requires_grad=False
                )
                vision_weight_list = vision_weight.split(".")
                target_module = self.vision_tower
                for nxt_module in vision_weight_list[:-1]:
                    target_module = getattr(target_module, nxt_module)
                setattr(target_module, vision_weight_list[-1], saved_weight)
            self.vision_tower = self.vision_tower.to(self.device)
            self.vision_tower.encoder.init_graph()

    def init_llm(self):
        self.language_model = FlashQwen2UsingMROPEForCausalLM(self.config.text_config, self.weights)

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
            **kwargs
    ):
        if is_prefill:
            if not torch.any(torch.eq(input_ids, self.image_token_id) | torch.eq(input_ids, self.video_token_id)):
                inputs_embeds = self.get_input_embeddings()(input_ids)
            else:
                inputs_embeds, image_grid_thw, video_grid_thw = self.prepare_prefill_token_service(input_ids)

                if image_grid_thw is not None or video_grid_thw is not None:
                    position_ids_thw_list = self._get_position_ids_thw(input_ids, position_ids, input_lengths,
                                                                       image_grid_thw, video_grid_thw)
                else:
                    position_ids_thw_list = []
                kwargs.update({"position_ids_thw_list": position_ids_thw_list})
                kwargs.update({"mrope_section": self.mrope_section})
        else:
            inputs_embeds = input_ids
        return self.language_model.forward(inputs_embeds,
                                           position_ids,
                                           is_prefill,
                                           kv_cache,
                                           block_tables,
                                           slots,
                                           input_lengths,
                                           max_seq_len,
                                           lm_head_indices,
                                           **kwargs)

    def prepare_prefill_token_service(self, input_ids):
        bos_pos = torch.where(torch.eq(input_ids, self.vision_start_token_id))[0]
        eos_pos = torch.where(torch.eq(input_ids, self.vision_end_token_id))[0]
        vision_num = bos_pos.shape[0]
        video_grid_thw = None
        image_grid_thw = None
        image_pixel_array = []
        video_pixel_array = []
        image_grid_thw_list = []
        video_grid_thw_list = []
        for i in range(vision_num):
            if input_ids[eos_pos[i] - 1] == self.image_token_id:
                image_thw_value = input_ids[bos_pos[i] + IMAGE_THW_TOKEN_OFFSET]
                image_grid_thw = torch.tensor(decode_shape_from_int64(image_thw_value), dtype=input_ids.dtype).npu()
                image_grid_thw_list.append(image_grid_thw)

                shm_value = input_ids[bos_pos[i] + SHM_VALUE_TOKEN_OFFSET]
                shape_value = input_ids[bos_pos[i] + SHAPE_VALUE_TOKEN_OFFSET]
                shared_array = get_data_from_shm(shm_value, shape_value, np.uint8, self.device)
                input_ids[bos_pos[i] + 1: bos_pos[i] + IMAGE_THW_TOKEN_OFFSET + 1] = self.image_token_id
                image_pixel_array.append(shared_array)
            elif input_ids[eos_pos[i] - 1] == self.video_token_id:
                video_thw_value = input_ids[bos_pos[i] + IMAGE_THW_TOKEN_OFFSET]
                video_grid_thw = torch.tensor(decode_shape_from_int64(video_thw_value), dtype=input_ids.dtype).npu()
                video_grid_thw_list.append(video_grid_thw)

                shm_value = input_ids[bos_pos[i] + SHM_VALUE_TOKEN_OFFSET]
                shape_value = input_ids[bos_pos[i] + SHAPE_VALUE_TOKEN_OFFSET]
                shared_array = get_data_from_shm(shm_value, shape_value, np.uint8, self.device)
                input_ids[bos_pos[i] + 1: bos_pos[i] + IMAGE_THW_TOKEN_OFFSET + 1] = self.video_token_id
                video_pixel_array.append(shared_array)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if image_pixel_array:
            image_grid_thw = torch.stack(image_grid_thw_list, dim=0)
            image_pixel = torch.cat(image_pixel_array)
            image_pixel = image_pixel.float().reshape(-1, NORMALIZATION_CHANNELS, PATCH_SZIE)
            image_pixel = self.normalizer(image_pixel)
            image_pixel = image_pixel.reshape(-1, NORMALIZATION_OUTPUT_SIZE)
            image_features = self.vision_tower(image_pixel.to(self.vision_tower.dtype), image_grid_thw)
            image_mask = input_ids == self.image_token_id
            inputs_embeds[image_mask] = image_features

        if video_pixel_array:
            video_grid_thw = torch.stack(video_grid_thw_list, dim=0)
            video_pixel = torch.cat(video_pixel_array)
            video_pixel = video_pixel.float().reshape(-1, NORMALIZATION_CHANNELS, PATCH_SZIE)
            video_pixel = self.normalizer(video_pixel)
            video_pixel = video_pixel.reshape(-1, NORMALIZATION_OUTPUT_SIZE)
            video_features = self.vision_tower(video_pixel.to(self.vision_tower.dtype), video_grid_thw)
            video_mask = input_ids == self.video_token_id
            inputs_embeds[video_mask] = video_features
        return inputs_embeds, image_grid_thw, video_grid_thw

    def _get_position_ids_thw(self, input_ids, position_ids, input_lengths, image_grid_thw, video_grid_thw):
        id_start = 0
        lengths_list = input_lengths.tolist()
        position_ids_thw_list = []
        image_num_before = 0
        video_num_before = 0
        for length in lengths_list:
            single_prefill_ids = input_ids[id_start:id_start + length]
            if not torch.any(torch.eq(single_prefill_ids, self.vision_start_token_id)):
                # 纯文本Batch
                single_position_ids = position_ids[id_start:id_start + length]
                position_ids_thw_list.append(single_position_ids.repeat(3, 1))
                continue
            vision_start_indices = torch.argwhere(single_prefill_ids == self.vision_start_token_id)
            vision_tokens = single_prefill_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == self.image_token_id).sum()
            video_nums = (vision_tokens == self.video_token_id).sum()

            single_image_grid_thw = \
                image_grid_thw[image_num_before:image_num_before + image_nums] if image_grid_thw is not None else None
            single_video_grid_thw = \
                video_grid_thw[video_num_before:video_num_before + video_nums] if video_grid_thw is not None else None
            image_num_before = image_num_before + image_nums
            video_num_before = video_num_before + video_nums

            position_ids_thw = self._get_rope_index(
                single_prefill_ids,
                single_image_grid_thw,
                single_video_grid_thw,
            )
            id_start = id_start + length
            position_ids_thw_list.append(position_ids_thw)
        return position_ids_thw_list

    def _get_rope_index(
            self,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor,
            video_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_index, video_index = 0, 0
        vision_start_indices = torch.argwhere(input_ids == self.vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == self.image_token_id).sum()
        video_nums = (vision_tokens == self.video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if self.image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(self.image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if self.video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(self.video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // self.spatial_merge_size,
                w.item() // self.spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        position_ids_thw = llm_positions.to(self.device).to(input_ids.dtype)
        return position_ids_thw
