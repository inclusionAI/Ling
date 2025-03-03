# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationConfig

from .input_builder import InputBuilder
from .model_utils import safe_get_tokenizer_from_pretrained, safe_get_config_dict
from .postprocessor import Postprocessor
from ...utils.env import ENV
from ...utils.log import logger


def remove_part_of_generation_config(generation_config):
    """Using the transformers' GenerationConfig class, update the generation configuration with the default value."""
    ori_gen = GenerationConfig()
    for key in generation_config:
        if key.endswith("_id"):
            continue
        ori_value = getattr(ori_gen, key, None)
        if ori_value is not None:
            generation_config[key] = ori_value
    return generation_config


@dataclass
class BaseRouter:
    """The base class of router.

    This class should be inherited by the corresponding router subclasses of specified models. A specified model can use
    a subclass router to find its custom properties.
    """
    model_name_or_path: str = ""

    config_dict: Any = None
    is_flash_causal_lm: bool = True
    load_tokenizer: bool = True
    max_position_embeddings: Optional[int] = None
    revision: Optional[str] = None
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    enable_atb_torch: bool = False

    # 初始化默认读取的autoconfig，各个模型可能会自定义，self.config会返回后续使用的config，注意不要循环依赖
    _config: Any = None
    _generation_config: Any = None
    _input_builder: Any = None
    _model_cls: Any = None
    _postprocessor: Any = None
    _tokenizer: Any = None
    is_inited: bool = False
    _tool_call_processor: Any = None

    def __post_init__(self):
        self.model_type = self.config_dict['model_type']
        if self.model_type == "chatglm" and "vision_config" in self.config_dict:
            self.model_type = "glm4v"
        if self.model_type == "internvl_chat":
            self.model_type = "internvl"
        if self.model_type == "llava_next_video":
            self.model_type = "llava_next"
        if self.model_type == "minicpmv" and "MiniCPM-Llama3-V-2_5" in self.model_name_or_path:
            self.model_type = "minicpm_llama3_v2"
        if self.model_type == "bunny-qwen2" or self.model_type == "bunny-minicpm":
            self.model_type = "bunny"
        if self.model_type == "bailing_moe":
            self.model_type = "deepseek"
        self.model_type_cap = self.model_type.capitalize()
        if self.model_type_cap == "Qwen2_moe" or self.model_type_cap == "Minicpm_llama3_v2":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Qwen2_audio":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Qwen2_vl":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_name_or_path

    @property
    def config(self):
        """The config property, which should not be overridden.

        It uses generation config to update config dictionary at first, and then uses the `get_config` method to get a
        config object. Note that the `get_config` method should use `config_dict` to initialize the config object.
        """
        if self._config is None:
            self._generation_config = self.generation_config
            if ENV.remove_generation_config_dict:
                self._generation_config = remove_part_of_generation_config(self._generation_config)
            self.config_dict.update(self._generation_config)
            self._config = self.get_config()
            if not hasattr(self._config, 'quantize'):
                setattr(self._config, 'quantize', None)
            if self.max_position_embeddings is not None:
                setattr(self._config, 'max_position_embeddings', self.max_position_embeddings)
        return self._config

    @property
    def generation_config(self):
        """The generation config property, which should not be overridden."""
        if self._generation_config is None:
            self._generation_config = self.get_generation_config()
        return self._generation_config

    @property
    def input_builder(self):
        """The input builder property, which should not be overridden."""
        if self._input_builder is None:
            self._input_builder = self.get_input_builder()
        return self._input_builder

    @property
    def model_cls(self):
        """The model class property, which should not be overridden."""
        if self._model_cls is None:
            self._model_cls = self.get_model_cls()
        return self._model_cls

    @property
    def model_version(self):
        """The model version property, which should not be overridden."""
        return ""

    @property
    def embedding_model_name(self):
        """The model name property, which should not be overridden."""
        return ""

    @property
    def postprocessor(self):
        """The postprocessor property, which should not be overridden."""
        if self._postprocessor is None:
            self._postprocessor = self.get_postprocessor()
        return self._postprocessor

    @property
    def tokenizer(self):
        """The tokenizer property, which should not be overridden."""
        if self._tokenizer is None and self.load_tokenizer:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    @property
    def toolscallprocessor(self):
        """The tools call processor property, which should not be overridden."""
        if self._tool_call_processor is None:
            self._tool_call_processor = self.get_toolscallprocessor()
        return self._tool_call_processor

    @staticmethod
    def check_config(config):
        """The validation of values in config."""
        eos_token_id_field = 'eos_token_id'

        vocab_size = 0
        vocab_size_field = 'vocab_size'
        if hasattr(config, vocab_size_field):
            vocab_size = getattr(config, vocab_size_field)
        attribute_ranges = {
            vocab_size_field: (1, 2147483647),
            'max_position_embeddings': (1, 2147483647),
            'hidden_size': (1, 2147483647),
            'intermediate_size': (1, 2147483647),
            'num_hidden_layers': (1, 1000),
            'num_attention_heads': (1, 10000),
            'initializer_range': (0, 2147483647),
            'rms_norm_eps': (0, 1),
            'pad_token_id': (-1, vocab_size),
            'bos_token_id': (0, vocab_size - 1),
            eos_token_id_field: (0, vocab_size - 1),
            'temperature': (0, 2),
            'top_k': (-1, vocab_size),
            'top_p': (0, 1),
            'repetition_penalty': (0, 2),
            'frequency_penalty': (-2, 2),
            'presence_penalty': (-2, 2)
        }
        if hasattr(config, "head_dim"):
            attribute_ranges['head_dim'] = (1, 1000)
        if hasattr(config, "num_key_value_heads"):
            attribute_ranges['num_key_value_heads'] = (1, 10000)

        def check_value(attr_ins, value_ins):
            if value_ins < min_val or value_ins > max_val:
                raise ValueError(f"self._config.{attr_ins} must be between {min_val} and {max_val}")

        def check_eos(eos_value):
            if isinstance(eos_value, int):
                check_value(eos_token_id_field, eos_value)
            elif isinstance(eos_value, list):
                for eos_v in eos_value:
                    if isinstance(eos_v, int):
                        check_value(eos_token_id_field, eos_v)
                    elif isinstance(eos_v, list):
                        for v in eos_v:
                            check_value(eos_token_id_field, v)
                    else:
                        raise ValueError("eos_token_id must be Union[int, List[Union[int, List[int]]]].")
            else:
                raise ValueError("eos_token_id must be Union[int, List[Union[int, List[int]]]].")

        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if attr == eos_token_id_field:
                check_eos(value)
                continue
            check_value(attr, value)

        if getattr(config, 'repetition_penalty', None) == 0:
            raise ValueError("repetition_penalty should not be 0.")
        if not isinstance(getattr(config, 'do_sample', None), bool):
            raise ValueError("do_sample must be bool.")

    def tokenize(self, inputs: List[Union[str, Dict[str, str]]], **kwargs) -> np.ndarray:
        """Transfer text input or multimodal input to token ids.

        Args:
            inputs: List | List[Dict], when it's List, it means the input for LLM.
                When it's List[Dict], it means the multimodal inputs in interleaved style,
                for example:
                    [
                        {'text': 'Let me show you two pictures'},
                        {'image': 'image_url_or_path'},
                        {'image': 'image_url_or_path'},
                        {'text': 'can you show the differences?'}
                    ]

        Returns:
            numpy.ndarray: The expanded input_ids whose dimension is 1.
        """
        return self.tokenizer([inputs[0]["text"]], return_tensors="np")["input_ids"][0]

    def get_config(self):
        """The default method to get config.

        A subclass router can override it to define a custom method getting config. Note that the `get_config` method
        should use `self.config_dict` instead of the model weight path to construct a config object.
        """
        try:
            config_cls = self.get_config_cls()
            config = config_cls.from_dict(self.config_dict)
        except Exception as e:
            logger.warning(str(e))
            config = PretrainedConfig.from_dict(self.config_dict)
        self.check_config(config)
        return config

    def get_generation_config(self):
        """The default method to get generation config."""
        generation_config_path = os.path.join(self.model_name_or_path, "generation_config.json")
        generation_config = {}
        if os.path.exists(generation_config_path):
            generation_config = safe_get_config_dict(generation_config_path)
        return generation_config

    def get_config_cls(self):
        """The default method to get config class."""
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + \
                                  f"{self.model_version}."
        config_file_name = f'config_{self.model_type}'
        module_path = f"{model_file_dir_name}{config_file_name}"
        module = importlib.import_module(module_path)
        config_cls_name = f"{self.model_type_cap}Config"
        return getattr(module, config_cls_name)

    def get_input_builder(self):
        """The default method to get input builder."""
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            return InputBuilder(self.tokenizer, max_length=self.config.max_position_embeddings)
        return InputBuilder(self.tokenizer)

    def get_model_cls(self):
        """The default method to get model class.

        This is a basic router method to find model class, which is usually not necessary to be overridden.
        """
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + \
                                  f"{self.model_version}."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        if self.embedding_model_name:  # for embedding model, example: gte-qwen2
            module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}_{self.embedding_model_name}"
        else:
            module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}"
        if self.enable_atb_torch:
            module_path += "_atb"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self.model_type_cap}ForCausalLM"
        if self.enable_atb_torch:
            model_cls_name += "ATB"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        return getattr(module, model_cls_name)

    def get_postprocessor(self):
        """The default method to get postprocessor."""
        return Postprocessor(self.tokenizer, self.generation_config)

    def get_tokenizer(self):
        """The default method to get tokenizer."""
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=True
        )

    def get_toolscallprocessor(self):
        """The default method to get tools call processor."""
        return ToolsCallProcessor(self.model_version)


class ToolsCallProcessor:
    """Base class for tools call processor."""
    def __init__(self, model_version):
        self.model_version = model_version

    @staticmethod
    def decode(content):
        """Parse model output to extract tools call output."""
        return content
