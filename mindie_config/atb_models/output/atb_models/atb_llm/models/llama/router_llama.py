# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from dataclasses import dataclass

from .config_llama import LlamaConfig
from .input_builder_llama import LlamaInputBuilder
from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .tool_call_process_llama import ToolsCallProcessorLlama


@dataclass
class LlamaRouter(BaseRouter):
    def get_config(self):
        config = LlamaConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        if self.config_dict['num_hidden_layers'] in [60]:
            # LLaMa 33B use_fast需要使用False
            use_fast = False
        else:
            use_fast = True
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=False,
            use_fast=use_fast
        )
        # 需要添加PAD token
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def get_input_builder(self):
        return LlamaInputBuilder(self.tokenizer, self.model_version)
    
    def get_toolscallprocessor(self):
        return ToolsCallProcessorLlama(self.model_version)
