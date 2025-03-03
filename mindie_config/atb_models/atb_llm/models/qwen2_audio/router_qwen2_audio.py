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
import os
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, List
import librosa
import torch
import numpy as np
from transformers import AutoProcessor
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm

from ..base.config import BaseConfig
from ..base.router import BaseRouter
from .flash_causal_qwen2_audio import Qwen2AudioConfig
from ..base.config import QuantizationConfig
from .data_process_qwen2_audio import load_feature_by_torchaudio
from .data_process_qwen2_audio import load_audio
from ..base.model_utils import safe_from_pretrained
from ..base.model_utils import safe_get_tokenizer_from_pretrained


SAMPLE_RATE = 16000


def from_list_format(list_format: List[Dict], image_token_str: str):
    text = ""
    num_images = 0
    for ele in list_format:
        if "audio" in ele or "image_or_video" in ele:
            num_images += 1
            text += "<|audio_bos|><|AUDIO|><|audio_eos|>"
        if "text" in ele:
            text += ele["text"]
    return text


def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def get_feature_lens(config: Any, inputs: list, processor: Any):
    image_or_video_path, text = None, None
    for k_v in inputs:
        if "text" in k_v:
            text = k_v["text"]
        if "audio" in k_v:
            image_or_video_path = k_v["audio"]
        if "image_or_video" in k_v:
            image_or_video_path = k_v["image_or_video"]

    image_or_video_token_id = config.audio_token_index
    audio = load_feature_by_torchaudio(image_or_video_path)

    prompt_head = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    text = prompt_head + text
    new_inputs = processor(text=text, audios=audio, return_tensors="pt")
    feature_attention_mask = new_inputs.feature_attention_mask

    _, audio_output_lengths = get_feat_extract_output_lengths(
        feature_attention_mask.sum(-1)
    )
    return [audio_output_lengths, image_or_video_token_id, [image_or_video_path], new_inputs]


def get_conversation_feature_lens(config: Any, inputs: list, processor: Any, text: str):
    audios, audio_paths = [], []
    audio_types = "audio_type"
    audio_type = audio_types.split('_')[0]
    for message in inputs:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if audio_type in ele:
                    audio_paths.append(ele[audio_type])
                    audio_byte = BytesIO(load_audio(ele[audio_type]))
                    audio_librosa = librosa.load(audio_byte, sr=SAMPLE_RATE)
                    audios.append(audio_librosa[0])

    image_or_video_token_id = config.audio_token_index
    new_inputs = processor(text=text, audios=audios, return_tensors="pt")
    feature_attention_mask = new_inputs.feature_attention_mask

    _, audio_output_lengths = get_feat_extract_output_lengths(
        feature_attention_mask.sum(-1)
    )
    return [audio_output_lengths, image_or_video_token_id, audio_paths, new_inputs]


@dataclass
class Qwen2Config(BaseConfig):
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    model_type = "qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        if "num_key_value_heads" not in kwargs:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class Qwen2audioRouter(BaseRouter):

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, "processor"):
            self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path)
        self.tokenizer.eos_token_id = self.config.text_config.eos_token_id

    @staticmethod
    def check_conversation(inputs):
        conversation_state = False
        for item in inputs:
            if 'role' in item:
                conversation_state = True
        return conversation_state
    
    @staticmethod
    def check_single_audio_file(inputs):
        for item in inputs:
            inputs_new = item['content'] if 'role' in item else inputs
        audio_num = 0
        for item in inputs_new:
            if 'audio' in item:
                audio_num += 1
                if audio_num > 1:
                    return False, inputs_new
        return True, inputs_new
    
    @staticmethod
    def process_shm(image_pixel, shm_name_save_path, dtype=np.float32):
        shm = create_shm(image_pixel.nbytes, shm_name_save_path)
        shared_array = np.ndarray(image_pixel.shape, dtype=dtype, buffer=shm.buf)
        shared_array[:] = image_pixel
        shm_name = encode_shm_name_to_int64(shm.name)
        shape_value = encode_shape_to_int64(image_pixel.shape)
        return shm_name, shape_value

    def tokenize(self, inputs, **kwargs):
        config = self.config
        processor = self.processor

        single_state, inputs_single = self.check_single_audio_file(inputs)
        if single_state:
            feature_lens, token_id, audio_paths, new_inputs = get_feature_lens(config, inputs_single, processor)
            image_or_video_token_str = processor.tokenizer.decode(token_id)
            prompt = from_list_format(inputs, image_or_video_token_str)
        else:
            if not self.check_conversation(inputs):
                inputs = [{"content": inputs, "role": "user"}]
            prompt = processor.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)
            conversation_data = get_conversation_feature_lens(config, inputs, processor, prompt)
            feature_lens, token_id, audio_paths, new_inputs = conversation_data
            image_or_video_token_str = processor.tokenizer.decode(token_id)

        pad_token_id = config.text_config.pad_token_id if config.text_config.pad_token_id is not None else -1

        pad_token_length = new_inputs.input_ids.size(1)
        new_inputs.feature_attention_mask = torch.tensor(new_inputs.feature_attention_mask, dtype=torch.int64)
        for feature_len in feature_lens:
            pad_token_length += feature_len - 1
        new_prompt_token = torch.full((pad_token_length, ), pad_token_id, dtype=torch.int64)

        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        shm_name_list, shape_value_list, audio_pixels = [], [], []

        audio_pixels.append(new_inputs.input_features)
        audio_pixels.append(new_inputs.input_ids)
        audio_pixels.append(new_inputs.attention_mask)
        audio_pixels.append(new_inputs.feature_attention_mask)
        for idx, audio_pixel in enumerate(audio_pixels):
            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(os.path.dirname(audio_paths[0]))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            if idx == 0:
                shm_name, shape_value = self.process_shm(audio_pixel, shm_name_save_path)
            else:
                shm_name, shape_value = self.process_shm(audio_pixel, shm_name_save_path, dtype=np.int64)
            shm_name_list.append(shm_name)
            shape_value_list.append(shape_value)
        
        for i, _ in enumerate(shm_name_list):
            new_prompt_token[2 * i] = torch.tensor(shm_name_list[i], dtype=torch.int64)
            new_prompt_token[2 * i + 1] = torch.tensor(shape_value_list[i], dtype=torch.int64)

        return new_prompt_token

    def get_config(self):
        config = safe_from_pretrained(Qwen2AudioConfig, self.model_name_or_path)
        config.model_name_or_path = self.model_name_or_path
        config.text_config = Qwen2Config.from_dict(config.text_config)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast,
        )

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        return position_ids

    def make_context(
        self, 
        rank: int,
        conversation: List[Dict[str, List[Dict]]], 
        system: str = "You are a helpful assistant.",
        **kwargs):
        context_tokens = self.tokenize(conversation)
        return context_tokens

    def get_input_builder(self):
        return self

