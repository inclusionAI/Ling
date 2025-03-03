# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from ..llama.config_llama import LlamaConfig


@dataclass
class ZiyaRouter(BaseRouter):
    def __post_init__(self):
        #ziya结构与llama结构相同
        super().__post_init__()
        self.model_type = "llama"
        self.model_type_cap = self.model_type.capitalize()
        
    def get_config(self):
        config = LlamaConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        if not self.is_flash_causal_lm:
            # FA需要添加PAD token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
