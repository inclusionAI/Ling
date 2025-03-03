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
from io import BytesIO
import numpy as np
import torchaudio
import librosa

from ...utils.file_utils import safe_open
from ...utils.multimodal_utils import safe_open_audio


SAMPLE_RATE = 16000


def load_audio(audio_path):
    try:
        with safe_open(audio_path, 'rb') as file:
            content = file.read()
    except Exception as e:
        raise FileNotFoundError("audio_path cannot open normally or is not exist.") from e
    return content


def prepare_conversation(conversation, audio, processor, is_server=False):
    audios, audios_path = [], []
    if is_server:
        audios_path = audio
        text = conversation
    else:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios_path.append(ele['audio_url'])
    for audio_path in audios_path:
        audios.append(
            librosa.load(
                BytesIO(load_audio(audio_path)),
                sr=processor.feature_extractor.sampling_rate)[0]
        )

    inputs = processor(text=text, audios=audios, return_tensors="pt")
    return inputs


def get_prefill_data(text, audio, processor):
    if isinstance(text, str) and isinstance(audio, str):
        audio, sr = safe_open_audio(torchaudio, audio)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = np.array(audio)[0]
        inputs = processor(text=text, audios=audio, return_tensors="pt")
    elif isinstance(text, list) and isinstance(text[0], dict):
        inputs = prepare_conversation(text, audio, processor)
    elif isinstance(text, str) and isinstance(audio, list):
        inputs = prepare_conversation(text, audio, processor, is_server=True)
    else:
        raise ValueError("The text should be str or list[dict].")
    return inputs


def load_feature_by_torchaudio(audio_path):
    try:
        audio, sr = safe_open_audio(torchaudio, audio_path)
    except FileNotFoundError as e:
        raise FileNotFoundError("audio_path is not correct, please check.") from e
    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)
    audio = np.array(audio)[0]
    return audio
