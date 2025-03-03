# Copyright (c) 2023; Baichuan Intelligent Technology
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.utils.log import logger
from ..base.input_builder import InputBuilder


class BaichuanInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, generation_config, **kwargs):
        self.model_version = model_version
        self.generation_config = generation_config
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, **kwargs):
        total_input, round_input = [], []
        for message in conversation[::-1]:
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.generation_config['user_token_id']] + content_tokens + round_input
                total_input = round_input + total_input
                round_input = []
            elif message['role'] == 'assistant':
                round_input = [
                                  self.generation_config['assistant_token_id']
                              ] + content_tokens + [
                                  self.generation_config['eos_token_id']
                              ] + round_input
            else:
                error_msg = f"message role not supported yet: {message['role']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        total_input.append(self.generation_config['assistant_token_id'])
        return total_input
