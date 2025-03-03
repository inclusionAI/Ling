# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.starcoder2.config_starcoder2 import Starcoder2Config
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


@dataclass
class Starcoder2Router(BaseRouter):
    def check_config_starcoder2(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attention_dropout': (0, 1),
            'residual_dropout': (0, 1),
            'embedding_dropout': (0, 1),
            'norm_epsilon': (0, 1),
            'num_key_value_heads': (1, 10000),
            'rope_theta': (0, 2147483647),
            'sliding_window': (0, 2147483647)
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
        config = Starcoder2Config.from_dict(self.config_dict)
        self.check_config_starcoder2(config)
        return config
