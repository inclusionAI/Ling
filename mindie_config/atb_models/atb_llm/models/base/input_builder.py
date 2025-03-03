# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from copy import deepcopy
from typing import Dict, Iterable, List

import numpy as np

from ...utils.log import logger, print_log


class InputBuilder:
    """Utility class used to construct the input.

    This class can be used to construct input ids for chat models and provide a custom interface for generating position
    ids for multimodal models.

    Args:
        tokenizer: The tokenizer used to tokenize the input.
        chat_template: The custom chat template which can replace the native tokenizer chat template.
        system_role_name: The role name of system. Default is 'system'.
        user_role_name: The role name of user. Default is 'user'.
        max_length: The maximum length of the input used to truncate the input of multi-turn conversations.
    """
    def __init__(self, tokenizer, chat_template="", system_role_name="system", user_role_name="user", max_length=2048):
        self.tokenizer = tokenizer
        self.system_role_name = system_role_name
        self.user_role_name = user_role_name
        self.tool_call_name = "tools_call"
        self.role_key = "role"
        if chat_template:
            self.tokenizer.chat_template = chat_template
        self.max_length = max_length
        self.rank = 0

    @staticmethod
    def generate_position_ids(input_ids: np.ndarray) -> Iterable:
        """A default method for generating position ids.

        This method can be overridden by subclasses to use custom position ids. For continuous batching, the caller will
        slice and pass the input ids of a single sequence into this method.

        Args:
            input_ids: The token ids with 1 dimension of only the input of a single sequence.
        """
        return range(len(input_ids))

    def make_context(self, rank: int, conversation: List[Dict[str, str]], add_generation_prompt: bool = True,
                     adapt_to_max_length: bool = False, **kwargs) -> List[int]:
        """Make chat context.

        It concatenates multi-turn conversation information into a complete input based on the chat template and convert
        it into a list of ids. If `adapt_to_max_length` is set to True, it will remove the oldest conversation turns to
        ensure that the concatenated list of token ids does not exceed the maximum length.

        Args:
            rank: The rank of the process.
            conversation: The list of dict containing conversations.
            add_generation_prompt: A boolean indicating whether to add a generation prompt.
            adapt_to_max_length: A boolean indicating whether to adapt to maximum length.
            **kwargs: Other arguments passed to the `_apply_chat_template` of this class or the subclasses.
        """
        self.rank = rank
        tools_msg = {}
        if len(conversation) != 0 and conversation[-1].get(self.role_key) == self.tool_call_name:
            tools_msg = conversation.pop()
        if adapt_to_max_length:
            conversation = deepcopy(conversation)
            if not conversation:
                raise ValueError("The conversation is empty!")
            system_turn = []
            if conversation[0].get(self.role_key) == self.system_role_name:
                system_turn = [conversation.pop(0)]
                if not conversation:
                    raise ValueError("There is not any queries in the conversation")
            query_turn = [conversation.pop()]
            if query_turn[0].get(self.role_key) != self.user_role_name:
                print_log(rank, logger.warning, f"The last query is not offered by {self.user_role_name}. "
                                                f"If your role name is not {self.user_role_name}, "
                                                f"please update it by user_role_name=`your_user_role_name`")
            last_turn = system_turn + query_turn
            last_turn_tokens = self._apply_chat_template(last_turn, add_generation_prompt=add_generation_prompt, 
                                                         **kwargs)
            num_turns = len(conversation)
            if len(last_turn_tokens) >= self.max_length:
                context_tokens = last_turn_tokens[:self.max_length]
                print_log(rank, logger.warning, f"The sentence `{context_tokens}` has been truncated "
                                                f"due to exceeding the maximum length limit.")
            elif num_turns <= 1:
                context_tokens = last_turn_tokens
            else:
                context_tokens = self._make_multi_turns_context(conversation, system_turn, query_turn, last_turn_tokens,
                                                                add_generation_prompt=add_generation_prompt, **kwargs)
        else:
            if len(tools_msg) != 0:
                context_tokens = self._apply_chat_template(conversation, add_generation_prompt=add_generation_prompt,
                                                        tools_msg=tools_msg, **kwargs)
            else:
                context_tokens = self._apply_chat_template(conversation, add_generation_prompt=add_generation_prompt,
                                                        **kwargs)

        print_log(rank, logger.debug, f"input_texts: {self.tokenizer.decode(context_tokens)}")
        return context_tokens

    def _apply_chat_template(self, conversation: List[Dict[str, str]], **kwargs) -> List[int]:
        """A basic application of chat template.

        This method should be overwritten if the model has special prompt template. Note that a parameter
        `add_generation_prompt=True` is passed through `kwargs` by default, which is supposed not influencing any
        template without this parameter. If you confirm that it affects your template, please filter it from `kwargs`.

        Args:
            conversation: A list contains dictionaries that must include `role` and `content`.
            **kwargs: Other arguments passed to the `apply_chat_template` method of tokenizer.

        Returns:
            List[int]: The token ids that have been concatenated with flags and tokenized.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Your transformers version is detected to be <4.34. This message indicates that this "
                               "model is not supported to run on transformers <4.34. You can upgrade transformers to "
                               "4.34 or above, or rewrite the InputBuilder provided with this model and load it in the "
                               "router.")
        if not self.tokenizer.chat_template:
            raise RuntimeError("The model does not appear to be a chat model because it is not configured with a "
                               "`chat_template`.")
        input_ids = self.tokenizer.apply_chat_template(conversation, **kwargs)
        return input_ids

    def _make_multi_turns_context(self, conversation, system_turn, query_turn, last_turn_tokens, **kwargs):
        content_key = "content"
        prompt, answer = conversation[-2], conversation[-1]
        multi_turns = system_turn + [prompt, answer] + query_turn
        multi_turns_tokens = self._apply_chat_template(multi_turns, **kwargs)
        one_turn_history_tokens = self.tokenizer.encode(prompt[content_key] + " " + answer[content_key])
        multi_turns_length = len(multi_turns_tokens)
        num_special = multi_turns_length - len(last_turn_tokens) - len(one_turn_history_tokens)
        print_log(self.rank, logger.info, f"The number of special tokens in one turn is: {num_special}")
        if multi_turns_length > self.max_length:
            return last_turn_tokens
        num_turns = len(conversation)
        if num_turns >= 4:
            window_index = num_turns - 2
            for i in range(num_turns - 4, -1, -2):
                prompt, answer = conversation[i], conversation[i + 1]
                one_turn_history_tokens = self.tokenizer.encode(prompt[content_key] + " " + answer[content_key])
                multi_turns_length += len(one_turn_history_tokens) + num_special
                if multi_turns_length > self.max_length:
                    break
                window_index = i
            if window_index != num_turns - 2:
                multi_turns = system_turn + conversation[window_index:] + query_turn
                multi_turns_tokens = self._apply_chat_template(multi_turns, **kwargs)
        return multi_turns_tokens
