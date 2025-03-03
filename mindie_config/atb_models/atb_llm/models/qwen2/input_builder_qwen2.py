# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Dict, Iterator, List
import copy

from .qwen_fncall_prompt import QwenFnCallPrompt
from .schema import Message, DEFAULT_SYSTEM_MESSAGE, SYSTEM, ASSISTANT, USER, ContentItem
from ..base.input_builder import InputBuilder


class Qwen2InputBuilder(InputBuilder):

    def __init__(self, tokenizer, **kwargs):
        self.content_key = "content"
        self.tools_key = "tools"
        self.role_key = "role"
        self.fncall_prompt = None
        super().__init__(tokenizer, **kwargs)

    def simulate_response_completion_with_chat(self, messages: List[Message]) -> List[Message]:
        if messages and (messages[-1].role == ASSISTANT):
            usr = messages[-2].content
            bot = messages[-1].content
            sep = '\n\n'
            if isinstance(usr, str) and isinstance(bot, str):
                usr = usr + sep + bot
            elif isinstance(usr, list) and isinstance(bot, list):
                usr = usr + [ContentItem(text=sep)] + bot
            else:
                raise NotImplementedError
            text_to_complete = copy.deepcopy(messages[-2])
            text_to_complete.content = usr
            messages = messages[:-2] + [text_to_complete]
        res = []
        for message in messages:
            message_dict = {self.role_key: message.role, self.content_key: message.content}
            res.append(message_dict)
        return res

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Your transformers version is detected to be <4.34. This message indicates that this "
                               "model is not supported to run on transformers <4.34. You can upgrade transformers to "
                               "4.34 or above, or rewrite the InputBuilder provided with this model and load it in the "
                               "router.")
        if not self.tokenizer.chat_template:
            raise RuntimeError("The model does not appear to be a chat model because it is not configured with a "
                               "`chat_template`.")
        roles = [conv.get('role') for conv in conversation]
        if tools_msg is None and 'tool' not in roles:
            input_ids = self.tokenizer.apply_chat_template(conversation, **kwargs)
        else:
            self.fncall_prompt = QwenFnCallPrompt()
            messages = []
            for msg in conversation:
                if isinstance(msg, dict):
                    messages.append(Message(**msg))
                else:
                    messages.append(msg)
            if messages[0]['role'] != SYSTEM:
                messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages
            if tools_msg is None:
                functions = None
                functions_choice = 'none'
            else:
                functions = tools_msg.get(self.tools_key)
                functions_choice = tools_msg.get('tool_choice')
            messages = self.fncall_prompt.preprocess_fncall_messages(
                messages=messages,
                functions=functions,
                lang='en',
                parallel_function_calls=False,
                function_choice=functions_choice
            )
            messages = self.simulate_response_completion_with_chat(messages)
            input_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return input_ids