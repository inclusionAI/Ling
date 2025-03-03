# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter
from .config_gpt_neox import GPTNeoXConfig


@dataclass
class Gpt_neoxRouter(BaseRouter):
    @staticmethod
    def check_config(config):
        attribute_ranges = {
            'layer_norm_eps': (0, 2147483647),
            'rotary_emb_base': (0, 2147483647),
            'rotary_pct': (0, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr):
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                error_msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(error_msg)
    
    
    def get_config(self):
        config = GPTNeoXConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.model_max_length = self.max_position_embeddings
        super().check_config(config)
        self.check_config(config)
        return config