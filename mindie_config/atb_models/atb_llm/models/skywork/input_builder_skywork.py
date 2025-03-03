# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from ..mixtral.input_builder_mixtral import MixtralInputBuilder


class SkyworkInputBuilder(MixtralInputBuilder):
    def __init__(self, tokenizer, model_version, **kwargs):
        super().__init__(tokenizer, model_version, **kwargs)