# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .config_telechat import TelechatConfig
from .input_builder_telechat import TelechatInputBuilder


@dataclass
class TelechatRouter(BaseRouter):
    def get_config(self):
        config = TelechatConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_input_builder(self):
        return TelechatInputBuilder(self.tokenizer, self.model_version, self.config, self.generation_config)
