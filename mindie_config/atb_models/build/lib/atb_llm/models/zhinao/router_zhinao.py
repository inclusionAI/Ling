# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from ..llama.router_llama import LlamaRouter


class ZhinaoRouter(LlamaRouter):
    def __post_init__(self):
        #zhinao360结构与llama结构相同
        super().__post_init__()
        self.model_type = "llama"
        self.model_type_cap = self.model_type.capitalize()
