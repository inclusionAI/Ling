# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .config_internlm import InternlmConfig


@dataclass
class InternlmRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称
        :return:
        """
        return "20b"

    def get_config(self):
        config = InternlmConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )
