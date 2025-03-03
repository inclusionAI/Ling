# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.starcoder.config_starcoder import StarcoderConfig
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


@dataclass
class StarcoderRouter(BaseRouter):
    def check_config_starcoder(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attn_pdrop': (0, 1),
            'embed_pdrop': (0, 1),
            'inference_runner': (0, 2147483647),
            'layer_norm_epsilon': (0, 1),
            'resid_pdrop': (0, 1),
            'summary_first_dropout': (0, 1)
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
    
    def get_config(self):
        config = StarcoderConfig.from_dict(self.config_dict)
        self.check_config_starcoder(config)
        return config 
