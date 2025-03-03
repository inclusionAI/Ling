# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
"""PyTorch MiniCPM_Llama3_V2 model."""

import importlib
import os
import json
from typing import Optional, List, Tuple
from copy import deepcopy
import numpy as np
from PIL import Image

from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
import torch

from atb_llm.models.base.flash_causal_multimodal import get_llm_model, MultiModalLLm
from ...utils.shm_utils import get_data_from_shm
from ...utils.multimodal_utils import safe_open_image
from ...utils.file_utils import safe_listdir, standardize_path, check_file_safety
from .resampler import Resampler


LLAMA = "llama"
MISTRAL = "mistral"
VICUNA = "vicuna"
MSG_CONTENT = "content"
INPUT_IDS = "input_ids"
IMAGE_BOS = 128010
IMAGE_PAD = 128002
EOS_TOKEN_ID = 128009


def process_qs(text, image):
    if isinstance(text, str):
        text = json.loads(text)
    copy_text = deepcopy(text)

    if len(text) == 0:
        raise RuntimeError("text is empty")

    if image is not None and isinstance(copy_text[0][MSG_CONTENT], str):
        copy_text[0][MSG_CONTENT] = [image, copy_text[0][MSG_CONTENT]]
    images = []
    for i, msg in enumerate(copy_text):
        role = msg["role"]
        content = msg[MSG_CONTENT]
        if role not in ["user", "assistant"]:
            raise RuntimeError("role must be user or assistant")
        if i == 0 and role != "user":
            raise RuntimeError("The role of first msg should be user")
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg[MSG_CONTENT] = "\n".join(cur_msgs)
    return copy_text, images



class FlashMinicpmllama3v2ForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        self.config = config
        if not config.quantize:
            setattr(config, 'quantize', None)
        else:
            setattr(config, 'quantize', config.quantize)
        super(MultiModalLLm, self).__init__(config, weights, **kwargs)
        self.weights = weights
        self.vision_tower = None
        self.language_model = None
        self.model_type = None
        self.init_multimodal()
        self.config.eos_token_id = EOS_TOKEN_ID
        self.vocab_size = config.vocab_size
        self.vision_dim = self.vision_tower.embed_dim
        self.embed_dim = self.config.hidden_size
        self.init_resampler(self.embed_dim, self.vision_dim)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.multi_modal_projector = None

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
            
    def init_vit(self):
        self.vision_tower = Idefics2VisionTransformer(self.config.vision_config)
        self.init_tower_weight(self.vision_tower, self.weights, "vpm")
        setattr(self.vision_tower, 'embed_dim', self.vision_tower.embeddings.embed_dim)
        setattr(self.vision_tower, 'patch_size', self.vision_tower.embeddings.patch_size)

    def init_llm(self):
        self.model_type = self.config.model_type
        if self.model_type in [MISTRAL, VICUNA, LLAMA]:
            self.model_type = LLAMA
        model_cls = get_llm_model("llama")
        self.config.pe_type = "ROPE"
        self.config.alibi_bias_max = None
        self.config.rope_keep_local_base_windows = None
        self.config.rope_vanilla_theta = None
        self.config.rope_mscale = None
        self.config.rope_given_inv_feq_str = None
        self.language_model = model_cls(self.config,
                                  self.weights,
                                  "llm.lm_head",
                                  "llm.model")
        self.language_model.skip_word_embedding = True

    def init_resampler(self, embed_dim, vision_dim):
        self.resampler = Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )
        self.init_resamplerweight(self.resampler, self.weights)

    def init_multimodal(self):
        self.init_vit()
        self.init_llm()
    
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
    
    def get_shm_data(self, input_ids, shm_index, name_idx, value_idx, dtype=np.float32):
        shm_name, shape_value = input_ids[shm_index[name_idx]], input_ids[shm_index[value_idx]]
        shared_tensor = get_data_from_shm(shm_name, shape_value, dtype, device=self.device)

        return shared_tensor

    def prefill_token_utils(self, inputs):
        if 'vision_hidden_states' not in inputs:
            dtype = self.config.torch_dtype
            tgt_sizes = inputs['tgt_sizes']
            pixel_values_list = inputs['pixel_values']
            vit_hidden_states = []
            all_pixel_values = []
            for pixel_values in pixel_values_list:
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            if all_pixel_values:
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                if self.config.batch_vision_input:
                    max_patches_size = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                    all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                       padding_value=0.0)
                    b_size, l_size, _ = all_pixel_values.shape
                    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(b_size, 3, -1, l_size)

                    patch_attn_mask = torch.zeros((b_size, 1, max_patches_size), dtype=torch.bool, device=self.device)
                    for i in range(b_size):
                        patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                    vision_embed = self.vision_tower(all_pixel_values.type(dtype),
                                                patch_attention_mask=patch_attn_mask).last_hidden_state
                    vision_embed = self.resampler(vision_embed, tgt_sizes)
                else:
                    # get vision_embed foreach
                    vision_embed = []
                    for single_tgt_size, single_pix_vals in zip(tgt_sizes, all_pixel_values):
                        single_pix_vals = single_pix_vals.unsqueeze(0)
                        b_size, l_size, _ = single_pix_vals.shape
                        single_pix_vals = single_pix_vals.permute(0, 2, 1).reshape(b_size, 3, -1, l_size)
                        single_vit_embed = self.vision_tower(single_pix_vals.type(dtype)).last_hidden_state
                        single_vit_embed = self.resampler(single_vit_embed, single_tgt_size.unsqueeze(0))
                        vision_embed.append(single_vit_embed)
                    vision_embed = torch.vstack(vision_embed)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vit_hidden_states.append(vision_embed[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vit_hidden_states.append([])
            else:  # no image
                dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vit_hidden_states.append(dummy_feature)
        else:
            vit_hidden_states = inputs['vision_hidden_states']

        if hasattr(self.language_model.config, 'scale_emb'):
            vllm_embed = (self.language_model.model.embed_tokens(inputs[INPUT_IDS])
                              * self.language_model.config.scale_emb)
        else:
            vllm_embed = self.language_model.model.embed_tokens(inputs[INPUT_IDS])

        vit_hidden_states = [i.type(vllm_embed.dtype) if isinstance(
            i, torch.Tensor) else i for i in vit_hidden_states]

        bs = len(inputs[INPUT_IDS])
        for idx in range(bs):
            cur_vs_hs = vit_hidden_states[idx]
            if len(cur_vs_hs) > 0:
                cur_vllm_embed = vllm_embed[idx]
                cur_image_bound = inputs['image_bound'][idx]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embed.device)

                    cur_vllm_embed.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_embed.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
        return vllm_embed[0]


    def prepare_prefill_token_service(self, input_ids):
        shm_index = torch.where(input_ids.lt(-1))[0]
        shm_length = shm_index.size(0) // 2
        if shm_length == 0:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            return inputs_embeds

        image_bound = self.get_shm_data(input_ids, shm_index, 0, 1, dtype=np.int64)
        tgt_sizes = self.get_shm_data(input_ids, shm_index, 2, 3, dtype=np.int64)

        image_pixels = []
        for i in range(2, shm_length):
            image_pixel = self.get_shm_data(input_ids, shm_index, 2 * i, 2 * i + 1)
            image_pixels.append(image_pixel)
        inputs = {}
        inputs['image_bound'] = [image_bound]
        inputs['tgt_sizes'] = [tgt_sizes]
        inputs['pixel_values'] = [image_pixels]
        for name_idx in range(shm_index.size(0)):
            input_ids[shm_index[name_idx]] = IMAGE_PAD
        inputs['input_ids'] = input_ids.unsqueeze(0)
        return self.prefill_token_utils(inputs)
        

    def prepare_prefill_token(self, multimodalinputs, processor, **kwargs):
        image = multimodalinputs.image
        text = multimodalinputs.text
        
        image = safe_open_image(Image, image)

        text = [text]
        copy_text, images = process_qs(text, image)
        system_prompt = ''
        if system_prompt:
            sys_msg = {'role': 'system', MSG_CONTENT: system_prompt}
            copy_text = [sys_msg] + copy_text

        prompt = processor.tokenizer.apply_chat_template(copy_text, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt").to(self.device)
        image.close()
        return self.prefill_token_utils(inputs)

    def init_ascend_operations(self, config: PretrainedConfig):
        pass

    def init_ascend_weight(self):
        pass
    
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, image_token_id):
        mask = (input_ids == image_token_id)
        inputs_embeds[mask] = image_features
        return inputs_embeds
