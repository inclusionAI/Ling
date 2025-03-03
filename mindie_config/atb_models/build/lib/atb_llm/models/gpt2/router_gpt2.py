# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .config_gpt2 import Gpt2Config

from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


@dataclass
class Gpt2Router(BaseRouter):
    def check_config_gpt2(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attn_pdrop': (0, 1),
            'embed_pdrop': (0, 1),
            'inference_runner': (0, 2147483647),
            'layer_norm_epsilon': (0, 1),
            'resid_pdrop': (0, 1),
            'summary_first_dropout': (0, 1),
            'n_embed': (1, 2147483647),
            'n_head': (1, 10000),
            'n_inner': (1, 2147483647),
            'n_layer': (1, 10000),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                logger.error(f"Error: self._config.{attr} is out of range",
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")
    
    def get_config(self):
        config = Gpt2Config.from_dict(self.config_dict)
        self.check_config_gpt2(config)
        return config 
