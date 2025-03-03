# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch qwen2_audio model."""
import numpy as np
from transformers import AutoProcessor
from transformers import Qwen2AudioEncoderConfig
import torch
from torch import nn
from ..base.flash_causal_multimodal import MultiModalConfig, MultiModalLLm
from .data_process_qwen2_audio import get_prefill_data
from ...utils.shm_utils import get_data_from_shm
from ..base.model_utils import safe_from_pretrained


class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.audio_config.d_model, config.text_config.hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class AudioInput:
    def __init__(self, input_features, attention_mask, input_ids, feature_attention_mask):
        self.input_features = input_features
        self.attention_mask = attention_mask
        self.input_ids = input_ids
        self.feature_attention_mask = feature_attention_mask


class Qwen2AudioConfig(MultiModalConfig):

    model_type = "qwen2_audio"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        audio_config=None,
        spatial_pool_mode="average",
        spatial_pool_stride=2,
        **kwargs,
    ):
        super().__init__(vision_config,
                         text_config,
                         audio_config,
                         **kwargs)
        self.text_config = text_config
        self.spatial_pool_mode = spatial_pool_mode
        self.spatial_pool_stride = spatial_pool_stride
        self._init_audioconfig(audio_config)

    def init_textconfig(self, text_config):
        self.text_config = text_config

    def _init_audioconfig(self, audio_config):
        self.audio_config = Qwen2AudioEncoderConfig()


class FlashQwen2audioForCausalLM(MultiModalLLm):
    def __init__(self,
                 config,
                 weights):
        kwargs = {"skip_word_embedding": True, "transformer_wte_parallel": False}
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.text_config.pad_token_id \
            if self.config.text_config.pad_token_id is not None else -1
        self.vision_resampler = None
        self.architectures = self.config.architectures
        self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path)
        self.init_multimodal()

    @staticmethod
    def init_multi_modal_projectorweight(module, weights):
        multimodel_weights = [multimodel_weight for multimodel_weight in module.state_dict().keys()]
        w_list = list(weights.routing.keys())
        for multimodel_weight in multimodel_weights:
            pop_index = w_list.index(f"multi_modal_projector.{multimodel_weight}")
            w_list.pop(pop_index)
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
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(self.config)
        self.init_multi_modal_projectorweight(self.multi_modal_projector, self.weights)

    def merge_input_ids_with_audio_features(self, audio_info, input_info):
        audio_features, num_audio_tokens = audio_info
        inputs_embeds, input_ids, attention_mask, labels = input_info

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, _ = input_ids.shape
        left_padding = True

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.config.audio_token_index) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), -1, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong."
                f" The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}."
                f" This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite

        return final_embedding, final_attention_mask, final_labels

    def get_input_embeddings(self):
        return self.language_model.transformer.wte
    
    def get_shm_data(self, input_ids, shm_index, name_idx, value_idx, dtype=np.float32):
        shm_name, shape_value = input_ids[shm_index[name_idx]], input_ids[shm_index[value_idx]]
        shared_tensor = get_data_from_shm(shm_name, shape_value, dtype, device=self.device)
        return shared_tensor

    def prepare_prefill_token_service(self, inputs_ids):
        shm_index = torch.where(inputs_ids.lt(-1))[0]
        shm_length = shm_index.size(0) // 2
        if shm_length == 0:
            inputs_embeds = self.get_input_embeddings()(inputs_ids)
            return inputs_embeds
        
        new_prompt_token = torch.tensor([], dtype=torch.bfloat16)
        for idx in range(shm_index.size(0) // 8):
            start = idx * 8
            input_features = self.get_shm_data(inputs_ids, shm_index, start, 1 + start)
            input_ids = self.get_shm_data(inputs_ids, shm_index, 2 + start, 3 + start, dtype=np.int64)
            attention_mask = self.get_shm_data(inputs_ids, shm_index, 4 + start, 5 + start, dtype=np.int64)
            feature_attention_mask = self.get_shm_data(inputs_ids, shm_index, 6 + start, 7 + start, dtype=np.int64)
            feature_attention_mask = torch.tensor(feature_attention_mask, dtype=torch.int32)
            inputs = AudioInput(input_features, attention_mask, input_ids, feature_attention_mask)
            inputs_embeds = self.prefill_token_utils(inputs)
            new_prompt_token = torch.cat([new_prompt_token, inputs_embeds])

        return new_prompt_token
    
    def prepare_prefill_token(self, multimodalparams, processor):
        text = multimodalparams.text
        audio = multimodalparams.audio
        inputs = get_prefill_data(text, audio, processor)
        inputs_embeds = self.prefill_token_utils(inputs)
        return inputs_embeds

    def prefill_token_utils(self, inputs):
        labels = None
        input_ids = inputs.input_ids.npu()
        attention_mask = inputs.attention_mask.npu()
        input_features = inputs.input_features.npu()
        feature_attention_mask = inputs.feature_attention_mask.npu()

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids.shape[1] != 1:
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand
            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
            # [1, 128, 3000] -->conv [1, 1280, 3000] -->conv [1, 1280, 1500] -->pool [1, 750, 1280]
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.multi_modal_projector(selected_audio_feature)

            audio_info = [audio_features, audio_output_lengths]
            input_info = [inputs_embeds, input_ids, attention_mask, labels]
            inputs_embeds, _, _ = self.merge_input_ids_with_audio_features(
                audio_info, input_info
            )
        inputs_embeds = inputs_embeds.view(inputs_embeds.shape[0] * inputs_embeds.shape[1],
                                            inputs_embeds.shape[2])

        return inputs_embeds
