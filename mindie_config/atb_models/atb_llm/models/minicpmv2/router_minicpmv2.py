# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass

from ..base.config import QuantizationConfig
from ..base.router import BaseRouter
from .configuration_minicpm import MiniCPMVConfig
from ..base.model_utils import safe_from_pretrained

from ...utils.log import logger
from ...utils.log.error_code import ErrorCode

EOT_ID = '<|eot_id|>'


@dataclass
class Minicpmv2Router(BaseRouter):
    def check_config_minicpmv2(self, config):
        super().check_config(config)
        attribute_ranges = {
            'mm_hidden_size': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                logger.error("error: value is out of range, please check: value", 
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_config(self):
        config = safe_from_pretrained(MiniCPMVConfig, self.model_name_or_path)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_minicpmv2(config)
        return config

    def get_generation_config(self):
        generation_config = super().get_generation_config()
        generation_config["eos_token_id"] = [
            generation_config["eos_token_id"],
            self.tokenizer.convert_tokens_to_ids(EOT_ID)
        ]
        return generation_config