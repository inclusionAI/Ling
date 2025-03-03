#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from typing import List
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.input_builder import InputBuilder


class MiniCpmInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, **kwargs):
        self.model_version = model_version
        super().__init__(tokenizer, **kwargs)

    def apply_chat_template_default(self, conversation, **kwargs):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        if conversation[0]["role"] == "system":
            conversation = [
                {
                    "role": conversation[1]["role"],
                    "content": B_SYS
                    + conversation[0]["content"]
                    + E_SYS
                    + conversation[1]["content"],
                }
            ] + conversation[2:]
        for i, msg in enumerate(conversation):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg["role"] != expected_role:
                msg = f"Invalid role at position {i}. Expected '{expected_role}', got '{msg['role']}'." + \
                      "model only supports 'system', 'user' and 'assistant' roles, " + \
                      "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                logger.error(
                    msg,
                    ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
                )
                raise ValueError(msg)
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(
                    conversation[::2],
                    conversation[1::2],
                )
            ],
            [],
        )
        if conversation[-1]["role"] != "user":
            msg = f"Last message must be from user, got {conversation[-1]['role']}"
            logger.error(
                msg,
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(msg)
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(conversation[-1]['content']).strip()} {E_INST}",
        )
        return dialog_tokens

    def _apply_chat_template(self, conversation, **kwargs):
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return super()._apply_chat_template(conversation, **kwargs)
        return self.apply_chat_template_default(conversation, **kwargs)