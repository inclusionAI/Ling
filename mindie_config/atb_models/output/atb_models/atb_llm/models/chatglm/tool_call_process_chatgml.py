# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import string
import random
import json
import ast


class ToolsCallProcessorChatglm:
    def __init__(self, model_version):
        self.model_version = model_version

    @staticmethod
    def parse_tool_call(tool_call_content):
        expr_ast = ast.parse(tool_call_content, mode='eval')
        node = expr_ast.body
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) :
                kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
                return kwargs
        raise ValueError("failed to parse tool calls.")

    @staticmethod
    def decode_v3_6b(text):
        tool_call_content = ""
        tool_call_list = []
        for response in text.split("<|assistant|>"):
            if len(response) == 0:
                continue
            metadata, tool_call_content = response.split("\n", maxsplit=1)
            tool_call_info = None
            if not metadata.strip():
                tool_call_content = tool_call_content.strip()
                tool_call_content = tool_call_content.replace(
                    "[[训练时间]]", "2023年")
            else:
                tool_call_content = "\n".join(
                    tool_call_content.split("\n")[1:2])
                try:
                    parameters = ToolsCallProcessorChatglm.parse_tool_call(tool_call_content)
                except Exception:
                    break
                if isinstance(parameters, dict):
                    tool_call_info = {
                        "name": metadata.strip(),
                        "arguments": json.dumps(parameters, ensure_ascii=False)
                    }
            # decode sucess
            if isinstance(tool_call_info, dict):
                characters = string.ascii_letters + string.digits
                call_id = "call_" + \
                    ''.join(random.choice(characters) for _ in range(8))
                call_res = {
                    "type": "function",
                    "id": call_id,
                    "function": tool_call_info
                }
                tool_call_list.append(call_res)

        if len(tool_call_list) != 0:
            return {
                "tool_calls": tool_call_list
            }
        return text

    @staticmethod
    def decode_v4_9b(content):
        lines = content.strip().split("\n")
        arguments_json = None

        if len(lines) >= 2 and lines[1].startswith("{"):
            function_name = lines[0].strip()
            arguments = "\n".join(lines[1:]).strip()
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
                call_id = "call_" + \
                    ''.join(random.choice(characters) for _ in range(8))
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
        if self.model_version == "v2_6b":
            return content
        elif self.model_version == "v3_6b":
            return self.decode_v3_6b(content)
        elif self.model_version == "v4_9b":
            return self.decode_v4_9b(content)
        return content
