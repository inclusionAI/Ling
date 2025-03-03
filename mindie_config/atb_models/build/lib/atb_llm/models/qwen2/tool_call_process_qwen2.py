# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import string
import random
import json

from .qwen_fncall_prompt import QwenFnCallPrompt


FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'


class ToolsCallProcessorQwen2:
    def __init__(self, model_version):
        self.model_version = model_version
        self.fncall_prompt = QwenFnCallPrompt()

    @staticmethod
    def decode_qwen(content):
        lines = content.strip()
        arguments_json = None
        is_tool_call = False
        if FN_NAME in lines and FN_ARGS in lines:
            arguments = lines.split(FN_ARGS)[1].split('✿')[0].strip(':').strip('\n').strip()
            function_name = lines.split(FN_NAME)[1].split('✿')[0].strip(':').strip('\n').strip()

            if function_name:
                is_tool_call = True
                try:
                    arguments_json = json.loads(arguments)
                except json.JSONDecodeError:
                    is_tool_call = False

            if is_tool_call:
                content = {
                    "name": function_name,
                    "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                            ensure_ascii=False)
                }
                characters = string.ascii_letters + string.digits
                call_id = "call_" + ''.join(random.choice(characters) for _ in range(8))
                call_res = {
                    "type": "function",
                    "id": call_id,
                    "function": content
                }
                return {
                    "tool_calls": [call_res]
                }
        return content.strip()

    def decode(self, content):
        return self.decode_qwen(content)
