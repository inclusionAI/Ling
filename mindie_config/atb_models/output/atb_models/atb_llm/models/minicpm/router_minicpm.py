#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from dataclasses import dataclass

from ..base.router import BaseRouter
from .modeling_minicpm import MiniCpmConfig
from .text_generator_minicpm import MiniCpmInputBuilder
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class MinicpmRouter(BaseRouter):
    def get_config(self):
        config = MiniCpmConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        super().check_config(config)
        return config

    def get_tokenizer(self):
        if self.config_dict['num_hidden_layers'] in [32]:
            use_fast = False
        else:
            use_fast = True
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )
        # 需要添加PAD token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
    
    def get_input_builder(self):
        return MiniCpmInputBuilder(self.tokenizer, self.model_version)
