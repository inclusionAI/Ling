# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class DeepseekInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, **kwargs):
        self.model_version = model_version
        super().__init__(tokenizer, **kwargs)

    def apply_chat_template_default(self, conversation, **kwargs):
        role_field = "role"
        content_field = "content"
        bos_token = "<｜begin▁of▁sentence｜>"
        eos_token = "<｜end▁of▁sentence｜>"
        formatted = bos_token
        for message in conversation:
            if message[role_field] == "user":
                formatted += "User: " + message[content_field] + "\n\n"
            elif message[role_field] == "assistant":
                formatted += "Assistant: " + message[content_field] + eos_token
            elif message[role_field] == "system":
                formatted += message[content_field] + "\n\n"
            else:
                msg = "Only user/assistant/system roles are supported!"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
        if "add_generation_prompt" in kwargs and kwargs["add_generation_prompt"]:
            formatted += "Assistant:"
        return self.tokenizer.encode(formatted, add_special_tokens=False)

    def _apply_chat_template(self, conversation, **kwargs):
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return super()._apply_chat_template(conversation, **kwargs)
        return self.apply_chat_template_default(conversation, **kwargs)
