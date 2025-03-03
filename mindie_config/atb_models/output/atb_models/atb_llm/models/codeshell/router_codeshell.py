# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter


@dataclass
class CodeshellRouter(BaseRouter):
    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        if self.max_position_embeddings:
            config.model_max_length = self.max_position_embeddings
        super().check_config(config)
        return config
