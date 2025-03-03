# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

""" InternLM-Xcomposer2 model configuration"""

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.config import BaseConfig


class Internlmxcomposer2Config(BaseConfig):
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rope_scaling_validation()
        if 'model_type' not in kwargs:
            self.model_type = 'internlmxcomposer2'
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if self.rope_scaling.type is None or self.rope_scaling.factor is None:
            msg = (
                "`rope_scaling` must have two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        rope_scaling_type = self.rope_scaling.type
        rope_scaling_factor = self.rope_scaling.factor
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            msg = f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor < 1.0:
            msg = f"`rope_scaling`'s factor field must be a float >= 1, got {rope_scaling_factor}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
