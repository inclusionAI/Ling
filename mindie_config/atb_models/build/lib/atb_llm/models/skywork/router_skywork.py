# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.models.mixtral.router_mixtral import MixtralRouter
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.skywork.flash_causal_skywork import SkyworkConfig
from atb_llm.models.skywork.input_builder_skywork import SkyworkInputBuilder


@dataclass
class SkyworkRouter(MixtralRouter):
    def get_config(self):
        config = SkyworkConfig.from_dict(self.config_dict)
        self.check_config_skywork(config)
        return config

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
            use_fast=True,
            pad_token='[PAD]'
        )

    def check_config_skywork(self, config):
        super().check_config(config)

    def get_input_builder(self):
        return SkyworkInputBuilder(self.tokenizer, self.model_version)