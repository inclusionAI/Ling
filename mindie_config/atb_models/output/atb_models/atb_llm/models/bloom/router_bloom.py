# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter


@dataclass
class BloomRouter(BaseRouter):
    def check_config_bloom(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attention_dropout': (0, 1),
            'hidden_dropout': (0, 1),
            'layer_norm_epsilon': (0, 1),
            'n_head': (1, 2147483647),
            'n_layer': (1, 2147483647),
            'offset_alibi': (1, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"The {attr} value in config must be between {min_val} and {max_val}"
                logger.error(
                    msg,
                    ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
                )
                raise ValueError(msg)

    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        if self.max_position_embeddings:
            config.seq_length = self.max_position_embeddings
            config.model_max_length = self.max_position_embeddings
            config.max_position_embeddings = self.max_position_embeddings
        super().check_config(config)
        self.check_config_bloom(config)
        return config

    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        if not self.is_flash_causal_lm:
            # FA需要添加PAD token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
