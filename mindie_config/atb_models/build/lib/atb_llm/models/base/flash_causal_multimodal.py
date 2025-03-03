# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
"""PyTorch  multimodal."""

from typing import Optional, List, Tuple
import abc
import importlib
import os
import torch
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import AutoModel
from transformers import AutoProcessor
from transformers.configuration_utils import PretrainedConfig
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from ..base.flash_causal_lm import FlashForCausalLM
from ..base.config import QuantizationConfig


MODEL_TYPE = "model_type"


def get_supported_models():
    """
    This method is used to retrieve all currently supported foundation models for Multimodal LLM.

    Returns:
        list: A list containing all supported foundation models.
    """
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_path = standardize_path(current_path)
    check_file_safety(current_path)
    supported_models = []
    for foldername in safe_listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)
    return supported_models


def get_llm_model(model_type):
    """
    Initializes the foundation model for the specified model type based on the input parameter.

    Args:
        model_type (str): The type of the model to initialize.

    Returns:
        object: The corresponding foundation model object.

    Raises:
        NotImplementedError: If the model does not exist.
        ImportError: If the model does cannot be automatically initialized.
    """
    supported_models = get_supported_models()
    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type};"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )

    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name = 'flash_causal'
    module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"无法导入模块{module_path}, 请确认路径是否正确"
        ) from e
    model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class MultiModalConfig(PretrainedConfig):
    """
    Creates a configuration for a multimodal large model.

    Args:
        vision_config (dict, optional): 
        Configuration for the vision encoder component of the multimodal model. Defaults to None.
        text_config (dict, optional): 
        Configuration for the foundation (LLM) component of the multimodal model. Defaults to None.
        audio_config (dict, optional): 
        Configuration for the audio encoder component of the multimodal model. Defaults to None.
    """
    def __init__(self, vision_config=None, text_config=None, audio_config=None, **kwargs):
        self.init_visionconfig(vision_config)
        self.init_textconfig(text_config)
        self.init_audioconfig(audio_config)
        super().__init__(**kwargs)
    
    @staticmethod
    def init_config(config):
        """
        Initializes the configuration.

        Args:
            config (dict or Config): The configuration dictionary or Config object.
        """
        if isinstance(config, dict):
            if MODEL_TYPE in config and config[MODEL_TYPE] in CONFIG_MAPPING:
                config = CONFIG_MAPPING[config[MODEL_TYPE]](**config)
        return config

    def init_visionconfig(self, vision_config):
        """
        Initializes the default vision configuration.

        Args:
            config (dict or Config): The configuration dictionary or Config object for the vision encoder.
        """
        self.vision_config = self.init_config(vision_config)

    def init_audioconfig(self, audio_config):
        """
        Initializes the default audio configuration.

        Args:
            config (dict or Config): The configuration dictionary or Config object for the audio encoder.
        """
        self.audio_config = self.init_config(audio_config)

    def init_textconfig(self, text_config):
        """
        Initializes the default LLM configuration.

        Args:
            config (dict or Config): The configuration dictionary or Config object for the LLM.
        """
        self.text_config = self.init_config(text_config)


class MultiModalLLm(FlashForCausalLM):
    """
    Creates an instance of a multimodal large model.

    Args:
        config (dict or Config): Configuration for the multimodal model.
        weights (str or bytes): Weights for the multimodal model.
    """
    def __init__(self, config, weights, **kwargs):
        if getattr(config, "text_config"):
            if not config.quantize:
                setattr(config.text_config, 'quantize', None)
            else:
                setattr(config.text_config, 'quantize', config.quantize)
            setattr(config.text_config, 'quantization_config', QuantizationConfig(**{}))
            super().__init__(config.text_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.kwargs = kwargs
        self.config = config
        self.weights = weights
        self.vocab_size = config.text_config.vocab_size
        self.vision_tower = None
        self.language_model = None
        self.audio_tower = None
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        if config.model_type != "yivl":
            self.processor = safe_from_pretrained(AutoProcessor, self.config.model_name_or_path,
                                                  trust_remote_code=self.trust_remote_code)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vision_prefix = kwargs.get("vision_prefix", "vision_tower")
        self.audio_prefix = kwargs.get("audio_prefix", "audio_tower")
        self.lmhead_prefix = kwargs.get("lmhead_prefix", "language_model.lm_head")
        self.model_prefix = kwargs.get("model_prefix", "language_model.model")
        if self.config.vision_config:
            self.init_vit()
        self.init_llm()
        if self.config.audio_config:
            self.init_audio()

        self.model_type = None

    @staticmethod
    def init_tower_weight(module, weights, prefix):
        """
        Initializes the weights of the specified encoder.

        Args:
            module (nn.Module): The encoder object.
            weights (Weight): The weights used to initialize the encoder.
            prefix (str): The prefix of the encoder's weights.
        """
        tower_weights = [tower_weight for tower_weight in module.state_dict().keys()]
        for tower_weight in tower_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"{prefix}.{tower_weight}"),
                    requires_grad=False
                )
            tower_weight_list = tower_weight.split(".")
            target_module = module
            for nxt_module in tower_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, tower_weight_list[-1], saved_weight)

    def init_vision_tower_weight(self, module, weights):
        """
        Initializes the weights of the vision encoder.

        Args:
            module (nn.Module): The vision encoder object.
            weights (Weight): The weights used to initialize the vision encoder.
            prefix (str): The prefix of the encoder's weights.
        """
        self.init_tower_weight(module, weights, self.vision_prefix)

    def init_audio_tower_weight(self, module, weights):
        """
        Initializes the weights of the audio encoder.

        Args:
            module (nn.Module): The audio encoder object.
            weights (Weight): The weights used to initialize the audio encoder.
        """
        self.init_tower_weight(module, weights, self.audio_prefix)

    def init_vit(self):
        """
        Initializes the visual encoder and load the weight.
        """
        self.vision_tower = AutoModel.from_config(self.config.vision_config)
        self.init_vision_tower_weight(self.vision_tower, self.weights)

    def init_llm(self):
        """
        Initializes the LLM.
        """
        self.model_type = self.config.text_config.model_type
        model_cls = get_llm_model(self.model_type)
        self.language_model = model_cls(self.config.text_config,
                                  self.weights,
                                  lmhead_prefix=self.lmhead_prefix,
                                  model_prefix=self.model_prefix,
                                  **self.kwargs)
        self.language_model.skip_word_embedding = True

    def init_audio(self):
        """
        Initializes the audio encoder and load the weight.
        """
        self.audio_tower = AutoModel.from_config(self.config.audio_config)
        self.init_audio_tower_weight(self.audio_tower, self.weights)

    @abc.abstractmethod
    def prepare_prefill_token_service(self, input_ids):
        """
        Handles the initial processing of the input ID passed to the service.
        """
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
                                          lm_head_indices)
