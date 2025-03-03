# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

""" Phi-3 model configuration"""

from typing import Optional

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.config import BaseConfig


class Phi3Config(BaseConfig):
    alibi_bias_max: float = 8.0
    bos_token_id: int = 1
    eos_token_id: int = 32000
    pad_token_id: int = 32000
    tie_word_embeddings: bool = False

    rope_given_inv_feq_str: Optional[str] = None
    rope_keep_local_base_windows: Optional[int] = None
    rope_mscale: Optional[int] = None
    rope_vanilla_theta: Optional[float] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rope_scaling_validation()
        if 'model_type' not in kwargs:
            self.model_type = 'phi3'
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

        if (self.rope_scaling.type is None
            or self.rope_scaling.long_factor is None
            or self.rope_scaling.short_factor is None):
            msg = (
                "`rope_scaling` must have three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        rope_scaling_type = self.rope_scaling.type
        rope_scaling_short_factor = self.rope_scaling.short_factor
        rope_scaling_long_factor = self.rope_scaling.long_factor
        if rope_scaling_type is None or rope_scaling_type not in ["su", "yarn"]:
            msg = f"`rope_scaling`'s type field must be one of ['su', 'yarn'], got {rope_scaling_type}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            msg = f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        if (
            not len(rope_scaling_short_factor)
            == self.hidden_size // self.num_attention_heads // 2
        ):
            msg = f"`rope_scaling`'s short_factor field must have length \
                    {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_short_factor)}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            msg = f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        if (
            not len(rope_scaling_long_factor)
            == self.hidden_size // self.num_attention_heads // 2
        ):
            msg = f"`rope_scaling`'s long_factor field must have length \
                    {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_long_factor)}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
