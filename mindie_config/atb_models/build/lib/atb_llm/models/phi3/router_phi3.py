# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.


from dataclasses import dataclass

from ..base.router import BaseRouter


@dataclass
class Phi3Router(BaseRouter):
    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        super().check_config(config)
        return config
