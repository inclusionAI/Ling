# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
from dataclasses import dataclass

from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class Internlmxcomposer2Router(BaseRouter):
    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )
        return tokenizer