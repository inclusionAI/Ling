# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import copy
from ..base.input_builder import InputBuilder


class TelechatInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, config, generation_config, **kwargs):
        self.model_version = model_version
        self.config = config
        self.generation_config = generation_config
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, **kwargs):
        history = copy.deepcopy(conversation)
        question = history.pop()
        q_token = self.tokenizer(question['content'])

        # get the max length we should build our inputs in
        model_max_length = self.config.seq_length
        if "max_new_tokens" in self.generation_config.keys():
            build_max_length = max(0, model_max_length - self.generation_config.max_new_tokens)
        else:
            build_max_length = max(0, self.generation_config['max_length'])
        if build_max_length < 3:
            raise ValueError("")

        user_id = self.generation_config['user_token_id']
        bot_id = self.generation_config['bot_token_id']
        eos_id = self.generation_config['eos_token_id']
        input_tokens = [user_id] + q_token['input_ids'][-build_max_length + 1:] + [bot_id]

        while len(history) > 0:
            message = history.pop()
            if message['role'] == self.user_role_name:
                tokens = [user_id] + self.tokenizer(message['content'])['input_ids']
            else:
                tokens = [bot_id] + self.tokenizer(message['content'])['input_ids'] + [eos_id]
            input_tokens = tokens + input_tokens
        return input_tokens
