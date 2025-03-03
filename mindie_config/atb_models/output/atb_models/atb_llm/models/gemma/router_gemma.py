# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .config_gemma import GemmaConfig


@dataclass
class GemmaRouter(BaseRouter):
    def get_config(self):
        config = GemmaConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config
