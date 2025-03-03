# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import string
import random
import json


class ToolsCallProcessorLlama:
    def __init__(self, model_version):
        self.model_version = model_version
        self.name_key = "name"
        self.param_key = "parameters"
        
    def decode(self, content):
        if not self.__content_is_ok(content):
            return content
        tool_call_list = []
        tool_call_content = json.loads(content)
        tool_call_detail = {
            self.name_key : tool_call_content.get(self.name_key),
            "arguments" : json.dumps(tool_call_content.get(self.param_key), ensure_ascii=False)
        }
        characters = string.ascii_letters + string.digits
        call_id = "call_" + \
            ''.join(random.choice(characters) for _ in range(8))
        call_res = {
            "type": "function",
            "id": call_id,
            "function": tool_call_detail
        }
        tool_call_list.append(call_res)
        return {"tool_calls": tool_call_list}

    def __content_is_ok(self, content: str):
        tool_call_content = ""
        try:
            tool_call_content = json.loads(content)
        except json.JSONDecodeError:
            return False
        if not isinstance(tool_call_content, dict):
            return False
        if (self.name_key not in tool_call_content) or (self.param_key not in tool_call_content):
            return False
        return True