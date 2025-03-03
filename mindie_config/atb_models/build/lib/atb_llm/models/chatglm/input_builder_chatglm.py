# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from ..base.input_builder import InputBuilder


class ChatglmInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, **kwargs):
        self.model_version = model_version
        self.content_key = "content"
        self.tools_key = "tools"
        self.role_key = "role"
        self.tool_calls_key = "tool_calls"
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        tools_msg = tools_msg if tools_msg is not None else {}
        if self.model_version == "v2_6b":
            if conversation[0][self.role_key] == self.system_role_name:
                conversation.pop(0)
            query = conversation.pop()["content"]
            prompt = ""
            for i in range(0, len(conversation), 2):
                old_query = conversation[i]
                response = conversation[i + 1]
                prompt += f"[Round {i + 1}]\n\n问：{old_query['content']}\n\n答：{response['content']}\n\n"
            prompt += f"[Round {len(conversation) + 1}]\n\n问：{query}\n\n答："
            tokens = self.tokenizer.encode(prompt)
        elif self.model_version == "v3_6b":
            last_message = conversation.pop()
            query = last_message.get(self.content_key, "")
            conversation = self._process_conversation(tools_msg, conversation)
            inputs = self.tokenizer.build_chat_input(
                query, history=conversation, role=self.user_role_name)
            tokens = inputs["input_ids"][0]
        else:
            conversation = self._process_conversation(tools_msg, conversation)
            tokens = super()._apply_chat_template(conversation, **kwargs)
        return tokens

    def _process_tool_choice(self, tools, tool_choice):
        processed_messages = []
        if self.model_version == "v4_9b":
            content = None
        elif self.model_version == "v3_6b":
            content = "Answer the following questions as best as you can. You have access to the following tools:"
        processed_messages.append(
            {
                self.role_key: "system",
                self.content_key: content,
                "tools": tools
            }
        )
        if isinstance(tool_choice, dict) and tools:
            choice_name = tool_choice["function"]["name"]
            if choice_name in [tool["function"]["name"] for tool in tools]:
                processed_messages.append(
                    {
                        self.role_key: "assistant",
                        "metadata": choice_name,
                        self.content_key: ""
                    }
                )
        return processed_messages

    def _process_message_item(self, role, content, tool_calls):
        if role == "tool":
            return {
                self.role_key: "observation",
                self.content_key: content,
                "function_call": True
            }
        elif role == "assistant":
            if tool_calls is not None and len(tool_calls) > 0:
                for tool_call in tool_calls:
                    func = tool_call.get("function", {})
                    return {
                        self.role_key: "assistant",
                        "metadata": func.get("name", ""),
                        self.content_key: func.get("arguments", "")
                    }
            else:
                if self.model_version == "v3_6b":
                    for response in content.split("<|assistant|>"):
                        metadata, _, sub_content = response.partition("\n")
                elif self.model_version == "v4_9b":
                    for response in content.split("\n"):
                        if "\n" in response:
                            metadata, _, sub_content = response.partition("\n")
                        else:
                            metadata, sub_content = "", response
                return {
                    self.role_key: role,
                    "metadata": metadata,
                    self.content_key: sub_content.strip()
                }
        else:
            return {self.role_key: role, self.content_key: content}
        
    def _process_conversation(self, tools_msg, conversation):
        processed_messages = []
        msg_has_sys = False
        tools = tools_msg.get("tools", None)
        tool_choice = tools_msg.get("tool_choice", None)
        if tools is not None:
            processed_messages.extend(self._process_tool_choice(tools, tool_choice))
            msg_has_sys = True

        for message in conversation:
            role = message.get(self.role_key, None)
            content = message.get(self.content_key, None)
            tool_calls = message.get(self.tool_calls_key, [])
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            message_item = self._process_message_item(
                role, content, tool_calls)
            processed_messages.append(message_item)

        return processed_messages
