# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Copyright 2023-present the HuggingFace Inc. team.
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
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from typing_extensions import Self

from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.log import logger, message_filter
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.quantize.quant_type import QUANTIZE_DESC_REQUIRED_LIST
from atb_llm.utils import file_utils


@dataclass
class RopeScaling:
    """A base class containing the scaling configuration for the RoPE embeddings.

    Attributes:
        factor:  In most scaling types, a factor of x will enable the model to handle
            sequences of length x original maximum pre-trained length.
        type: The sub-variant of RoPE to use.
        long_factor: The scaling factor to be applied to long contexts.
        short_factor: The scaling factor to be applied to short contexts.
        low_freq_factor: Only used with llama3. Scaling factor applied to low frequency components of the RoPE.
        high_freq_factor: Only used with llama3. Scaling factor applied to high frequency components of the RoPE.
        original_max_position_embeddings: The original max position embeddings used during pretraining.
        rope_type: The sub-variant of RoPE to use.
        rope_theta: The base period of the RoPE embeddings.
        attention_factor: The scaling factor to be applied on the attention computation.
        beta_fast: Only Used with `type` equals to `yarn`.
            Parameter to set the boundary for extrapolation (only) in the linear ramp function.
        beta_slow: Only Used with `type` equals to `yarn`.
            Parameter to set the boundary for extrapolation (only) in the linear ramp function
        rotary_emb_base: The base period of the RoPE embeddings.
    """
    factor: float = 1.0
    type: str = 'linear'

    long_factor: Optional[Any] = None
    short_factor: Optional[Any] = None

    # rope param for new feature
    low_freq_factor: Optional[Any] = None
    high_freq_factor: Optional[Any] = None
    original_max_position_embeddings: Optional[Any] = None
    rope_type: Optional[Any] = None
    rope_theta: Optional[Any] = None
    attention_factor: Optional[Any] = None
    beta_fast: Optional[int] = 32
    beta_slow: Optional[int] = 1
    rotary_emb_base: Optional[Any] = None

    
@dataclass
class QuantizationConfig:
    """A base class used to store model quantization information.

    Attributes:
        group_size: group size used in per-group quantization.
        kv_quant_type: type for key-value cache quantization.
        fa_quant_type: type for flash attention quantization.
        reduce_quant_type: type for all reduce quantization.
    """
    group_size: int = 0
    kv_quant_type: Optional[str] = None
    fa_quant_type: Optional[str] = None
    reduce_quant_type: Optional[str] = None

    def __init__(self, group_size=0, kv_quant_type=None, fa_quant_type=None, reduce_quant_type=None, **kwargs):
        self.group_size = group_size
        self.kv_quant_type = kv_quant_type
        self.fa_quant_type = fa_quant_type
        self.reduce_quant_type = reduce_quant_type

    def to_dict(self):
        """Convert the object's attributes to a dictionary"""
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class BaseConfig(PretrainedConfig):
    """A base class used to store model configuration information.

    This class defines some common fields that will be called by higher-level components. Each model should implement a
    subclass that inherits from this class to initialize its specific configuration. This approach ensures that the
    fields called by higher-level components are guaranteed to exist in the configuration object's attributes, thereby
    preventing exceptions during calls. Additionally, this base class performs security checks on common parameters,
    eliminating the need for each model to implement its own security validation functions.
    Note: Whether using the base class or subclass, the `from_dict` method should be used to construct the object.
    Otherwise, parameter validation will be bypassed, potentially leaving security risks caused by invalid parameters.
    The `from_pretrained` method in this class is only used to perform security validation on the model_path passed in
    case the method is called unexpectedly. This helps prevent security risks caused by tampered weight files. Under no
    circumstances should `from_pretrained` be considered the primary method to construct a configuration object.

    Attributes:
        max_position_embeddings: The maximum number of position encodings, typically used to represent the maximum
            number of tokens the model can accept. Some models also refer to this as `max_sequence_length`.
        vocab_size: Vocabulary size, which represents the number of token ids that the model can recognize.
        quantize: A string that indicates the quantization type of the model's weights.
        rope_scaling: A dict or an object of `RopeScaling` class of rope scaling configuration information, detailed in
            the `RopeScaling` class. The default value is None.
        quantization_config: A dict or an object of `QuantizationConfig` class of quantization configuration
            information, detailed in the `QuantizationConfig` class. The default value is None.
    """
    max_position_embeddings: int = 0
    vocab_size: int = 0

    quantize: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any] | RopeScaling] = None
    quantization_config: Optional[Dict[str, Any] | QuantizationConfig] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parse_obj()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> 'BaseConfig':
        """Method used to construct an object of this class.

        This method is used to construct an object from a dictionary, which allows additional parameters to be passed
        via kwargs.

        Args:
            config_dict: A dictionary containing configuration parameters.
            **kwargs: Additional configuration parameters which override the configuration parameters in `config_dict`.
        """
        config_dict.update(kwargs)
        config = cls(**config_dict)
        if 'top_k' not in config_dict:
            setattr(config, 'top_k', 1000)
        config.validate()
        return config
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'BaseConfig':
        """An alternative constructor method.

        This method only performs a security check on the model weight path and skips parameter validation. When
        `from_dict` is available, please avoid using this method to construct an instance of this class.

        Args:
            model_path: Path to the model weight file.
            **kwargs: Additional keyword arguments passed to the `from_pretrained` method.
        """
        model_path = file_utils.standardize_path(model_path, check_link=False)
        file_utils.check_path_permission(model_path)
        try:
            config = super().from_pretrained(model_path, local_files_only=True, **kwargs)
        except EnvironmentError:
            raise EnvironmentError("Get config from pretrained failed, "
                                   "please check config file in the model path.") from None
        except Exception:
            raise ValueError("Get config from pretrained failed, "
                             "please check the input parameters model_path.") from None
        return config

    def parse_obj(self):
        """Construct `rope_scaling` and `quantization_config` as objects based on the dictionary information."""
        if self.rope_scaling is None:
            self.rope_scaling = {}
        self.rope_scaling = RopeScaling(**self.rope_scaling)

        if self.quantization_config is None:
            self.quantization_config = {}
        self.quantization_config = QuantizationConfig(**self.quantization_config)

    def validate(self):
        """Perform security validation on the input parameters."""
        self.__check_max_position_embeddings()
        self.__check_vocab_size()
        self.__check_rope_scaling()
        self.__check_group_size()
        self.__check_quantize()
        self.__check_kv_quant_type()
        self.__check_reduce_quant_type()
        self.__check_rotary_emb_base()
        self.__check_rope_scaling_type()
        self.__check_rope_scaling_factor()
        self.__check_long_factor()
        self.__check_short_factor()
        self.__check_low_freq_factor()
        self.__check_high_freq_factor()
        self.__check_original_max_position_embeddings()
        self.__check_rope_theta()
        self.__check_attention_factor()
        self.__check_beta_slow()
        self.__check_beta_fast()
        self.__check_rope_type()
        self.__check_fa_quant_type()

    def __check_max_position_embeddings(self):
        if self.max_position_embeddings == 0:
            raise ValueError('The parameter `max_position_embeddings` is not initialized.')

    def __check_vocab_size(self):
        if self.vocab_size == 0:
            raise ValueError('The parameter `vocab_size` is not initialized.')

    def __check_rope_scaling(self):
        if self.rope_scaling is not None:
            if not isinstance(self.rope_scaling.factor, float) or self.rope_scaling.factor < 1.0:
                error_msg = f"`rope_scaling`'s factor field must be a float >= 1, got {self.rope_scaling.factor}"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)
            if self.rope_scaling.type not in ["linear", "dynamic", "su", "yarn"]:
                error_msg = "`rope_scaling`'s type field must be one of " \
                    "['linear', 'dynamic', 'su', 'yarn']"
                raise ValueError(error_msg)

    def __check_group_size(self):
        if self.quantization_config.group_size not in [0, 64, 128]:
            error_msg = f"Unsupported value: {self.quantization_config.group_size}, " \
                f"this field is only supported for value 0, 64, or 128, " \
                f"loaded from config.json -> quantization_config -> group_size."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)
    
    def __check_quantize(self):
        quantize_type = [item.value for item in QUANTIZE_DESC_REQUIRED_LIST]
        if self.quantize is not None and self.quantize not in quantize_type:
            error_msg = f"`quantize` is only supported for value in {quantize_type}, " \
                f"loaded from config.json -> quantize. " \
                f"The specific types supported by each model are different, please refer to the model README file."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)

    def __check_kv_quant_type(self):
        if self.quantization_config.kv_quant_type is not None and self.quantization_config.kv_quant_type not in ['C8']:
            error_msg = "`kv_quant_type` is only supported for type `C8`, " \
                "loaded from config.json -> quantization_config -> kv_quant_type. " \
                "The specific types supported by each model are different, please refer to the model README file."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)
    
    def __check_reduce_quant_type(self):
        if self.quantization_config.reduce_quant_type is not None and \
            self.quantization_config.reduce_quant_type not in ['per_channel']:
            error_msg = "`reduce_quant_type` is only supported for type `per_channel`, " \
                "loaded from config.json -> quantization_config -> reduce_quant_type. " \
                "The specific types supported by each model are different, please refer to the model README file."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)

    def __check_rope_scaling_type(self):
        if self.rope_scaling.type is not None and self.rope_scaling.type not in ['linear', 'dynamic', 'yarn', 'llama3']:
            error_msg = "rope_scaling_type is only supported for 'linear', 'dynamic', 'yarn', 'llama3' " \
                        "loaded from config.json -> rope_scaling -> type. " \
                        "The specific ranges supported by each model are different, " \
                        "please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_rope_scaling_factor(self):
        if self.rope_scaling.factor is not None and not (-65504 <= self.rope_scaling.factor <= 65504):
            error_msg = f"Unsupported range: {self.rope_scaling.factor}, " \
                        f"this field is only supported for range [-65504,65504], " \
                        f"loaded from config.json -> rope_scaling -> factor. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_short_factor(self):
        if self.rope_scaling.short_factor is not None and not (-65504 <= self.rope_scaling.short_factor <= 65504):
            error_msg = f"Unsupported range: {self.rope_scaling.short_factor}, " \
                        f"this field is only supported for range [-65504,65504], " \
                        f"loaded from config.json -> rope_scaling -> short_factor. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_long_factor(self):
        if self.rope_scaling.long_factor is not None and not (-65504 <= self.rope_scaling.long_factor <= 65504):
            error_msg = f"Unsupported range: {self.rope_scaling.long_factor}, " \
                        f"this field is only supported for range [-65504,65504], " \
                        f"loaded from config.json -> rope_scaling -> long_factor. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_rotary_emb_base(self):
        if self.rope_scaling.rotary_emb_base is not None and not (1 <= self.rope_scaling.factor <= 2147483647):
            error_msg = f"Unsupported range: {self.rope_scaling.rotary_emb_base}, " \
                        f"this field is only supported for range [1,2147483647], " \
                        f"loaded from config.json -> rope_scaling -> rotary_emb_base. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_low_freq_factor(self):
        if self.rope_scaling.low_freq_factor is not None and not (1 <= self.rope_scaling.low_freq_factor <= 4):
            error_msg = f"Unsupported range: {self.rope_scaling.low_freq_factor}, " \
                        f"this field is only supported for range [1,4], " \
                        f"loaded from config.json -> rope_scaling -> low_freq_factor. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_high_freq_factor(self):
        if self.rope_scaling.high_freq_factor is not None and not (1 <= self.rope_scaling.high_freq_factor <= 4):
            error_msg = f"Unsupported range: {self.rope_scaling.high_freq_factor}, " \
                        f"this field is only supported for range [1,4], " \
                        f"loaded from config.json -> rope_scaling -> high_freq_factor. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_original_max_position_embeddings(self):
        if (self.rope_scaling.original_max_position_embeddings is not None and
                not (1 <= self.rope_scaling.original_max_position_embeddings <= 2147483647)):
            error_msg = f"Unsupported ranges: {self.rope_scaling.original_max_position_embeddings}, " \
                        f"this field is only supported for range [1,2147483647], " \
                        f"loaded from config.json -> rope_scaling -> original_max_position_embeddings. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_rope_theta(self):
        if self.rope_scaling.rope_theta is not None and not (1 <= self.rope_scaling.rope_theta <= 2147483647):
            error_msg = f"Unsupported ranges: {self.rope_scaling.rope_theta}, " \
                        f"this field is only supported for range [1,2147483647], " \
                        f"loaded from config.json -> rope_scaling -> rope_theta. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_attention_factor(self):
        if (self.rope_scaling.attention_factor is not None and
                not (1 <= self.rope_scaling.attention_factor <= 2147483647)):
            error_msg = f"Unsupported ranges: {self.rope_scaling.attention_factor}, " \
                        f"this field is only supported for range [0,1], " \
                        f"loaded from config.json -> rope_scaling -> attention_factor. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_beta_fast(self):
        if self.rope_scaling.beta_fast is not None and not (1 <= self.rope_scaling.beta_fast <= 2147483647):
            error_msg = f"Unsupported ranges: {self.rope_scaling.beta_fast}, " \
                        f"this field is only supported for range [1,2147483647], " \
                        f"loaded from config.json -> rope_scaling -> beta_fast. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_beta_slow(self):
        if self.rope_scaling.beta_slow is not None and not (1 <= self.rope_scaling.beta_slow <= 2147483647):
            error_msg = f"Unsupported ranges: {self.rope_scaling.beta_slow}, " \
                        f"this field is only supported for range [1,2147483647], " \
                        f"loaded from config.json -> rope_scaling -> beta_slow. " \
                        f"The specific ranges supported by each model are different, " \
                        f"please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)

    def __check_rope_type(self):
        if (self.rope_scaling.rope_type is not None and
                self.rope_scaling.rope_type not in ["dynamic", "yarn", "llama3"]):
            error_msg = "rope_scaling.rope_type is only supported for type `dynamic`, `yarn`, `llama3`, " \
                        "loaded from config.json -> rope_scaling -> rope_type. " \
                        "The specific types supported by each model are different," \
                        " please refer to the model README file."
            error_msg = message_filter(error_msg)
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)


    def __check_fa_quant_type(self):
        if self.quantization_config.fa_quant_type is not None and \
            self.quantization_config.fa_quant_type not in ['FAQuant']:
            error_msg = "`fa_quant_type` is only supported for type `FAQuant`, " \
                "loaded from config.json -> quantization_config -> fa_quant_type. " \
                "The specific types supported by each model are different, please refer to the model README file."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)


@dataclass
class LoraConfig:
    """A base class used to store lora configuration information.

    Attributes:
        r: Lora attention dimension.
        lora_alpha: Lora alpha.
        use_rslora: When set to True,  which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it
            was proven to work better. Otherwise, it will use the original default value of `lora_alpha/r`.
        rank_pattern: The mapping from layer names to ranks which are different from the default rank specified by `r`.
            For example, `{'model.decoder.layers.0.encoder_attn.k_proj': 8}`.
        alpha_pattern: The mapping from layer names to alphas which are different from
            the default alpha specified by `lora_alpha`.
            For example, `{'model.decoder.layers.0.encoder_attn.k_proj': 32}`
    """
    r: int = field(default=8)
    lora_alpha: int = field(default=8)
    use_rslora: bool = field(default=False)
    rank_pattern: Optional[dict] = field(default_factory=dict)
    alpha_pattern: Optional[dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> Self:
        """Method used to construct an object of LoraConfig class.

        This method is used to construct an object from a dictionary, which allows additional parameters to be passed
        via kwargs.

        Args:
            config_dict: A dictionary containing configuration parameters.
            **kwargs: Additional configuration parameters which override the configuration parameters in `config_dict`.
        """
        config_dict.update(kwargs)
        field_names = set(f.name for f in dataclasses.fields(cls))
        config = cls(**{k: v for k, v in config_dict.items() if k in field_names})
        config.validate()
        return config

    def validate(self):
        """Perform security validation on the input parameters."""
        self.__check_r()
        self.__check_lora_alpha()
        self.__check_rank_pattern()
        self.__check_alpha_pattern()

    def __check_r(self):
        if self.r <= 0 or self.r > 1024:
            raise ValueError('The parameter `r` must be larger than 0 and less than or equal to 1024')

    def __check_lora_alpha(self):
        if self.lora_alpha <= 0 or self.lora_alpha > 1024:
            raise ValueError('The parameter `lora_alpha` must be larger than 0 and less than or equal to 1024')

    def __check_rank_pattern(self):
        if self.rank_pattern is None:
            return
        if not isinstance(self.rank_pattern, dict):
            raise ValueError('The parameter `rank_pattern` must be a dictionary')
        for rank_key, rank_value in self.rank_pattern.items():
            if rank_value <= 0 or rank_value > 1024:
                error_msg = f'The parameter `{rank_key}` must be larger than 0 and less than or equal to 1024'
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)

    def __check_alpha_pattern(self):
        if self.alpha_pattern is None:
            return
        if not isinstance(self.alpha_pattern, dict):
            raise ValueError('The parameter `alpha_pattern` must be a dictionary')
        for alpha_key, alpha_value in self.alpha_pattern.items():
            if alpha_value <= 0 or alpha_value > 1024:
                error_msg = f'The parameter `{alpha_key}` must be larger than 0 and less than or equal to 1024'
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)
