# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import dataclasses
from typing import List

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image_placeholder>"
SEP_TOKEN = "###"


@dataclasses.dataclass
class Conversation:
    system: str    
    roles: List[str]
    messages: List[List[str]]
    sep: str = SEP_TOKEN

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and isinstance(messages[0][1], tuple):
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image_placeholder>", "").strip()
            messages[0] = (init_role, "<image_placeholder>\n" + init_msg)

        ret = self.system + "\n\n" + self.sep + " "
        for role, message in messages:
            if message:
                if isinstance(message, tuple):
                    message, _, _ = message
                ret += role + ": " + message + "\n" + self.sep + " "
            else:
                ret += role + ":"
        return ret
    
    def append_message(self, role, message):
        self.messages.append([role, message])


def render_text(input_question):
    input_question = DEFAULT_IMAGE_TOKEN + "\n" + input_question
    conversation = Conversation(
        system=("This is a chat between an inquisitive human and an AI assistant. "
                "Assume the role of the AI assistant. Read all the images carefully, "
                "and respond to the human's questions with informative, helpful, detailed and polite answers. "
                "这是一个好奇的人类和一个人工智能助手之间的对话。"
                "假设你扮演这个AI助手的角色。仔细阅读所有的图像，"
                "并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。"),
        roles=("Human", "Assistant"),
        messages=[],
        sep="###",
    )
    conversation.append_message(conversation.roles[0], input_question)
    conversation.append_message(conversation.roles[1], None)

    return conversation.get_prompt()


def insert_separator(input_strs, sep):
    ans = []
    for sublist in zip(input_strs, [sep] * len(input_strs)):
        for ele in sublist:
            ans.append(ele)
    return ans


def tokenize_text(input_text, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    prompt = render_text(input_text)
    prompt_chunk_ids = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    input_ids = []
    offset = 0
    if (len(prompt_chunk_ids) > 0
        and len(prompt_chunk_ids[0]) > 0
        and prompt_chunk_ids[0][0] == tokenizer.bos_token_id):
        offset = 1
        input_ids.append(prompt_chunk_ids[0][0])

    for x in insert_separator(prompt_chunk_ids, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return input_ids[:-1]