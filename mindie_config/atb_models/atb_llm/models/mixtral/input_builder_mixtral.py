# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.input_builder import InputBuilder


class MixtralInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, **kwargs):
        self.model_version = model_version
        super().__init__(tokenizer, **kwargs)

    def apply_chat_template_default(self, conversation, **kwargs):
        role_field = "role"
        content_field = "content"
        bos_token = "<s>"
        eos_token = "</s>"
        system_message = ""
        if conversation[0][role_field] == "system":
            system_message = conversation[0][content_field]
            conversation = conversation[1:]
        formatted = bos_token
        for idx, message in enumerate(conversation):
            if (message[role_field] == "user") != (idx % 2 == 0):
                msg = "After the optional system message, " \
                      "conversation roles must alternate user/assistant/user/assistant/..."
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)
            if message[role_field] == "user":
                if (idx == 0) and system_message:
                    formatted += " [INST] " + system_message + "\n\n" + message[content_field] + " [/INST]"
                else:
                    formatted += " [INST] " + message[content_field] + " [/INST]"
            elif message[role_field] == "assistant":
                formatted += " " + message[content_field] + eos_token
            else:
                msg = "Only user and assistant roles are supported, " \
                      "with the exception of an initial optional system message!"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)
        return self.tokenizer.encode(formatted, add_special_tokens=False)

    def _apply_chat_template(self, conversation, **kwargs):
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return super()._apply_chat_template(conversation, **kwargs)
        return self.apply_chat_template_default(conversation, **kwargs)
