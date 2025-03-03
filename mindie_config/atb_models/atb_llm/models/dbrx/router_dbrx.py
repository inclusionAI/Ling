# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .configuration_dbrx import DbrxConfig
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class DbrxRouter(BaseRouter):
    @property
    def config(self):
        return DbrxConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
        )

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            padding_side="left",
            trust_remote_code=False,
            use_fast=True,
            pad_token='[PAD]'
        )
        return tokenizer
