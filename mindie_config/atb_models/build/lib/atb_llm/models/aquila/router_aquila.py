# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class AquilaRouter(BaseRouter):

    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            pad_token='<|endoftext|>',
            padding_side="left",
            trust_remote_code=False,
            use_fast=True
        )
        if not self.is_flash_causal_lm:
            # FA需要添加PAD token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
