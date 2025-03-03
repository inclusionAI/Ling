# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .flash_causal_mixtral import MixtralConfig
from .input_builder_mixtral import MixtralInputBuilder
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


@dataclass
class MixtralRouter(BaseRouter):
    def get_config(self):
        config = MixtralConfig.from_dict(self.config_dict)
        self.check_config_mixtral(config)
        return config

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
            use_fast=True,
            pad_token='[PAD]'
        )

    def check_config_mixtral(self, config):
        super().check_config(config)
        attribute_ranges = {
            "attention_dropout" : (0, 2147483647), 
            "initializer_range" : (0, 2147483647),
            "num_attention_heads" : (0, 2147483647),
            "num_experts_per_tok" : (1, 8),
            "num_key_value_heads" : (1, 2147483647),
            "n_routed_experts " : (2, 8),
            "rope_theta": (0, 2147483647), 
            "router_aux_loss_coef": (0, 2147483647) 
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
            
        if getattr(config, "num_experts_per_tok", 0) > getattr(config, "n_routed_experts", 0):
            msg = "self._config.num_experts_per_tok must be smaller than or equal to self._config.n_routed_experts"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        sliding_window = getattr(config, "sliding_window")
        if sliding_window and (sliding_window < 0 or sliding_window > 2147483647):
            msg = "self._config.sliding_window must be between 0 and 2147483647"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        
        hidden_act = getattr(config, "hidden_act")
        if hidden_act != "silu":
            msg = "self._config.hidden_act must be silu"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if not isinstance(getattr(config, "use_cache", False), bool):
            msg = "self._config.use_cache must be a boolean"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        
        if not isinstance(getattr(config, "output_router_logits", False), bool):
            msg = "self._config.output_router_logits must be a boolean"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if not isinstance(getattr(config, "tie_word_embeddings", False), bool):
            msg = "self._config.tie_word_embeddings must be a boolean"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

    def get_input_builder(self):
        return MixtralInputBuilder(self.tokenizer, self.model_version)