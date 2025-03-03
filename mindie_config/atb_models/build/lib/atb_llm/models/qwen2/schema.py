# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, model_validator

DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

ROLE = 'role'
CONTENT = 'content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'
TOOL = 'tool'


class BaseModelCompatibleDict(BaseModel):

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return f'{self.model_dump()}'

    def model_dump(self, **kwargs):
        return super().model_dump(exclude_none=True, **kwargs)

    def model_dump_json(self, **kwargs):
        return super().model_dump_json(exclude_none=True, **kwargs)

    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default


class Function(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    def __repr__(self):
        return f'Function({self.model_dump()})'


class ToolCall(BaseModelCompatibleDict):
    function: Function

    def __init__(self, function: Function):
        super().__init__(function=function)

    def __repr__(self):
        return f'ToolCall({self.model_dump()})'


class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None

    def __init__(self, text: Optional[str] = None, image: Optional[str] = None, file: Optional[str] = None):
        super().__init__(text=text, image=image, file=file)

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

    @property
    def type(self) -> Literal['text', 'image', 'file']:
        t, v = self.get_type_and_value()
        return t

    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v

    @model_validator(mode='after')
    def check_exclusivity(self):
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', or 'file' must be provided.")
        return self

    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file'], str]:
        (t, v), = self.model_dump().items()
        return t, v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
    function_call: Optional[Function] = None
    tool_calls: Optional[List[ToolCall]]
    extra: Optional[dict] = None

    def __init__(self,
                 role: str,
                 content: Optional[Union[str, List[ContentItem]]] = None,
                 name: Optional[str] = None,
                 function_call: Optional[Function] = None,
                 tool_calls: Optional[ToolCall] = None,
                 extra: Optional[dict] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role, content=content,
                         name=name, function_call=function_call, tool_calls=tool_calls, extra=extra)

    def __repr__(self):
        return f'Message({self.model_dump()})'
