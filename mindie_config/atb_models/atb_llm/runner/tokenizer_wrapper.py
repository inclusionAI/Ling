# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from ..models import get_model


class TokenizerWrapper:
    """A class for the upper layer to call the model's customized tokenizer.

    This class provides objects such as the model's configuration, tokenizer, and input builder, etc. The
    `input_builder` can assemble the prompt according to the chat template, and is a core function of the chat service
    interface.

    Args:
        model_name_or_path: The model weight path or model identifier.
    """

    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        router_ins = get_model(model_name_or_path, **kwargs)
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.postprocessor = router_ins.postprocessor
        self.tokenize = router_ins.tokenize
        self.toolscallprocessor = router_ins.toolscallprocessor
