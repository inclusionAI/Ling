# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import copy
import json
from typing import Dict, List, Literal, Union

from .schema import ASSISTANT, TOOL, SYSTEM, USER, ContentItem, Message


AUTO = 'auto'
ZH = 'zh'
EN = 'en'


class QwenFnCallPrompt(object):

    @staticmethod
    def preprocess_fncall_messages(
        messages: List[Message],
        functions: List[dict],
        lang: Literal[EN, ZH],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal[AUTO], str] = AUTO,
    ) -> List[Message]:
        ori_messages = messages

        # Change function_call responses to plaintext responses:
        messages = []
        for msg in copy.deepcopy(ori_messages):
            role, content = msg.role, msg.content
            if role in (SYSTEM, USER):
                messages.append(msg)
            elif role == ASSISTANT:
                fn_call = msg.tool_calls[0]
                if fn_call:
                    f_name = fn_call.function.name
                    f_args = fn_call.function.arguments
                    if f_args.startswith('```'):  # if code snippet
                        f_args = '\n' + f_args  # for markdown rendering
                    func_content = '\n' if messages[-1].role == ASSISTANT else ''
                    func_content += f'{FN_NAME}: {f_name}'
                    func_content += f'\n{FN_ARGS}: {f_args}'
                    content += func_content
                if messages[-1].role == ASSISTANT:
                    messages[-1].content += content
                else:
                    messages.append(Message(role=role, content=content))
            elif role == TOOL:
                if content:
                    f_result = copy.deepcopy(content)
                else:
                    f_result = ''
                f_exit = f'\n{FN_EXIT}: '
                last_text_content = messages[-1].content
                if last_text_content.endswith(f_exit):
                    messages[-1].content = last_text_content[:-len(f_exit)]
                f_result = f'\n{FN_RESULT}: ' + f_result + f_exit
                messages[-1].content += f_result
            else:
                raise TypeError

        # Add a system prompt for function calling:
        if functions is not None:
            tool_desc_template = FN_CALL_TEMPLATE[lang + ('_parallel' if parallel_function_calls else '')]
            tool_descs = '\n\n'.join(get_function_description(function, lang=lang) for function in functions)
            tool_names = ','.join(function['function'].get('name_for_model', function['function'].get('name', '')) \
                                  for function in functions)
            tool_system = tool_desc_template.format(tool_descs=tool_descs, tool_names=tool_names)
            if messages[0].role == SYSTEM:
                messages[0].content = messages[0].content + '\n\n' + tool_system
            else:
                messages = [Message(role=SYSTEM, content=[ContentItem(text=tool_system)])] + messages

        # Remove ': ' for continued generation of function calling,
        # because ': ' may form a single token with its following words:
        if messages[-1].role == ASSISTANT:
            last_msg = messages[-1].content
            if last_msg.endswith(f'{FN_EXIT}: '):
                messages[-1].content = messages[-1].content[:-2]

        # Add the function_choice prefix:
        if function_choice not in (AUTO, 'none'):
            if messages[-1].role == ASSISTANT:
                last_msg = messages[-1]
                if last_msg.content:
                    if extract_text_from_message(last_msg).endswith(FN_EXIT):
                        last_msg.content += ': \n'
                    else:
                        last_msg.content += '\n'
                messages = messages[:-1]
            else:
                last_msg = Message(role=ASSISTANT, content='')
            last_msg.content = f'{FN_NAME}: {function_choice}'
            messages = messages + [last_msg]
        return messages


FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'

FN_STOP_WORDS = [FN_RESULT, FN_EXIT]

FN_CALL_TEMPLATE_INFO_ZH = """# 工具

## 你拥有如下工具：

{tool_descs}"""

FN_CALL_TEMPLATE_INFO_EN = """# Tools

## You have access to the following tools:

{tool_descs}"""

FN_CALL_TEMPLATE_FMT_ZH = """## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

%s: 工具名称，必须是[{tool_names}]之一。
%s: 工具输入
%s: 工具结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs:

%s: The input of the tool
%s: The tool to use, should be one of [{tool_names}]
%s: Tool results
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (

    FN_ARGS,
    FN_NAME,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_ZH = """## 你可以在回复中插入以下命令以并行调用N个工具：

%s: 工具1的名称，必须是[{tool_names}]之一
%s: 工具1的输入
%s: 工具2的名称
%s: 工具2的输入
...
%s: 工具N的名称
%s: 工具N的输入
%s: 工具1的结果
%s: 工具2的结果
...
%s: 工具N的结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_EN = """## Insert the following command in your reply when you need \
to call N tools in parallel:

%s: The name of tool 1, should be one of [{tool_names}]
%s: The input of tool 1
%s: The name of tool 2
%s: The input of tool 2
...
%s: The name of tool N
%s: The input of tool N
%s: The result of tool 1
%s: The result of tool 2
...
%s: The result of tool N
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE = {
    'zh': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_ZH,
    'en': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_EN,
    'zh_parallel': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_ZH,
    'en_parallel': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_EN,
}


def get_function_description(function: Dict, lang: Literal[EN, ZH]) -> str:
    """
    Text description of function
    """
    function = function['function']
    tool_desc_template = {
        ZH: '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
        EN: '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
    }
    tool_desc = tool_desc_template.get(lang)
    name = function.get('name', None)
    name_for_human = function.get('name_for_human', name)
    name_for_model = function.get('name_for_model', name)

    if name_for_model == 'code_interpreter':
        args_format = {
            ZH: '此工具的输入应为Markdown代码块。',
            EN: 'Enclose the code within triple backticks (`) at the beginning and end of the code.',
        }
    else:
        args_format = {
            ZH: '此工具的输入应为JSON对象。',
            EN: 'Format the arguments as a JSON object.',
        }
    args_format = function.get('args_format', args_format.get(lang))

    return tool_desc.format(name_for_human=name_for_human,
                            name_for_model=name_for_model,
                            description_for_model=function['description'],
                            parameters=json.dumps(function['parameters'], ensure_ascii=False),
                            args_format=args_format).rstrip()


def extract_text_from_message(msg: Message,) -> str:
    if isinstance(msg.content, str):
        text = msg.content
    else:
        raise TypeError(f'List of str or str expected, but received {type(msg.content).__name__}.')
    return text.strip()



