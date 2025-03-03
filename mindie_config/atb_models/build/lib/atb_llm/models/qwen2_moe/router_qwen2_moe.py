# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from dataclasses import dataclass
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from ..base.router import BaseRouter
from .configuration_qwen2_moe import Qwen2MoeConfig


@dataclass
class Qwen2moeRouter(BaseRouter):
    @property
    def config(self):
        return Qwen2MoeConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
        )

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=self.trust_remote_code
        )

