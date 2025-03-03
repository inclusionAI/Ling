# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from ..deepseek.router_deepseek import DeepseekRouter


class Bailing_moeRouter(DeepseekRouter):
    def __post_init__(self):
        #zhinao360结构与llama结构相同
        super().__post_init__()
        self.model_type = "deepseek"
        self.model_type_cap = self.model_type.capitalize()

