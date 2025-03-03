# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass

from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .flash_causal_deepseekv2 import DeepseekV2Config

from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


@dataclass
class Deepseekv2Router(BaseRouter):
    def get_config(self):
        config = DeepseekV2Config.from_pretrained(self.model_name_or_path)
        self.check_config_deepseekv2(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def check_config_deepseekv2(self, config):
        super().check_config(config)

        # 校验rope_scaling参数数目，需要等于7
        if not isinstance(config.rope_scaling_dict, dict) or len(config.rope_scaling_dict) != 7:
            msg = "`rope_scaling` must be a dictionary with with 7 fields, " \
                  f"got {len(config.rope_scaling_dict)}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)


        rope_scaling_type = config.rope_scaling_dict.get("type", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "yarn"]:
            msg = f"`rope_scaling`'s type field must be one of ['linear', 'yarn'], got {rope_scaling_type}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if config.topk_method not in ["greedy", "group_limited_greedy"]:
            msg = f"`topk_method`'s type field must be one of ['greedy', 'group_limited_greedy'], " \
                  f"got {config.topk_method}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        # 校验topk参数是否匹配
        if config.topk_method == "greedy" and config.topk_group != config.n_group and config.n_group != 1:
            msg = f"`topk_method is `greedy`, please set `topk_group=1` and `n_group=1`, " \
                  f"got topk_group={config.topk_group}, n_group={config.n_group}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)