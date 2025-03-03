# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import List
from ..base.input_builder import InputBuilder

from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


class LlamaInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, **kwargs):
        self.model_version = model_version
        super().__init__(tokenizer, **kwargs)

    def apply_chat_template_default(self, conversation, **kwargs):
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        content = "content"
        role = "role"
        if conversation[0][role] == "system":
            conversation = [
                {
                    role: conversation[1][role],
                    content: b_sys
                    + conversation[0][content]
                    + e_sys
                    + conversation[1][content],
                }
            ] + conversation[2:]
        for i, msg in enumerate(conversation):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg[role] != expected_role:
                logger.error("error: Invalid role",
                             ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(f"Invalid role at position {i}. Expected '{expected_role}', got '{msg[role]}'."
                                  "model only supports 'system', 'user' and 'assistant' roles, "
                                  "starting with 'system', then 'user' and alternating (u/a/u/a/u...)")
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{b_inst} {(prompt[content]).strip()} {e_inst} {(answer[content]).strip()} ",
                )
                for prompt, answer in zip(conversation[::2], conversation[1::2])
            ],
            [],
        )
        if conversation[-1][role] != "user":
            logger.error("error: Last message must be from user", 
                         ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(f"Last message must be from user, got {conversation[-1][role]}")
        dialog_tokens += self.tokenizer.encode(
            f"{b_inst} {(conversation[-1][content]).strip()} {e_inst}",
        )
        return dialog_tokens

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            if tools_msg:
                tools_list = tools_msg.get("tools", None)
                # tools call need transformers>=4.42.0
                return super()._apply_chat_template(conversation, tools=tools_list, **kwargs)
            return super()._apply_chat_template(conversation, **kwargs)
        return self.apply_chat_template_default(conversation, **kwargs)
