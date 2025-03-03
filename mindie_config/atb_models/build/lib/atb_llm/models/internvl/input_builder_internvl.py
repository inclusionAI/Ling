# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# --------------------------------------------------------
# Copyright (c) 2024 OpenGVLab
# --------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, List

import torch

from fastchat.conversation import Conversation, register_conv_template, SeparatorStyle
from fastchat.conversation import conv_templates

from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.models.internvl.data_preprocess_internvl import process_image_input, process_video_input

from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger


_CONTENT = "content"
_IMAGE = "image"
_VIDEO = "video"
_TEXT = "text"


INTERNVL_SYSTEM_PROMPTS = {
    'v1': {
        'Hermes-2': 'Answer the questions.',
        'internlm2-chat': 'You are an AI assistant whose name is InternLM (书生·浦语).',
        'phi3-chat': 'You are an AI assistant whose name is Phi-3.'
    },
    'v2': {
        'Hermes-2': (
            '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，'
            '英文名叫InternVL, 是一个有用无害的人工智能助手。'
        ),
        'internlm2-chat': (
            '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，'
            '英文名叫InternVL, 是一个有用无害的人工智能助手。'
        ),
        'phi3-chat': (
            '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，'
            '英文名叫InternVL, 是一个有用无害的人工智能助手。'
        ),
        'internvl2_5': (
            '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学'
            '及多家合作单位联合开发的多模态大语言模型。'
        ),
    }
}


class InternvlInputBuilder(InputBuilder):
    def __init__(self, tokenizer, config, **kwargs):
        super().__init__(tokenizer, system_role_name="assistant", user_role_name="user", **kwargs)
        self.config = config
        self.template = get_internvl_conv_template(config.template)
        self.image_size = self.config.force_image_size or self.config.vision_config.image_size
        self.patch_size = self.config.vision_config.patch_size
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.config.downsample_ratio ** 2))
        self.use_dynamic_prepro = False if config.ps_version == "v1" else True
        self.img_begin_id = self.tokenizer.encode("<img>")[-1]
        self.img_end_id = self.tokenizer.encode("</img>")[-1]

    def make_context(
        self, 
        rank: int,
        conversation: List[Dict[str, List[Dict]]], 
        add_generation_prompt: bool = True,
        adapt_to_max_length: bool = False, 
        **kwargs):
        if not isinstance(conversation[0]["content"], list):
            logger.error("The conversation \"content\" should be a List[Dict].",
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise TypeError("The conversation \"content\" should be a List[Dict].")
        
        shm_name_save_path = kwargs.get('shm_name_save_path', None)

        context_tokens = self._apply_chat_template(
            conversation,
            shm_name_save_path=shm_name_save_path,
            )
        return context_tokens

    def _apply_chat_template(
        self,
        conversation: List[Dict[str, List[Dict]]],
        shm_name_save_path: str = None,
        **kwargs):

        text = ""
        image_index = 1
        shm_name_list = []
        shape_value_list = []
        image_num = sum(1 for d in conversation if _IMAGE in d)
        for message in conversation:
            query = ""
            for single_input in message[_CONTENT]:
                if _TEXT in single_input:
                    text += single_input.get(_TEXT)
                    continue
                if _IMAGE in single_input:
                    current_query, shm_name_value, shape_value = process_image_input(
                        single_input,
                        image_num,
                        image_index,
                        self.use_dynamic_prepro,
                        self.num_image_token,
                        shm_name_save_path
                    )
                    query += current_query
                    image_index += 1
                    shm_name_list.append(shm_name_value)
                    shape_value_list.append(shape_value)
                elif _VIDEO in single_input:
                    current_query, shm_name_value, shape_value = process_video_input(
                        single_input,
                        self.use_dynamic_prepro,
                        self.num_image_token,
                        shm_name_save_path
                    )
                    query += current_query
                    shm_name_list += shm_name_value
                    shape_value_list += shape_value
                else:
                    logger.error("Unsupport element, it should be a video or image, please check your input",
                                 ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                    raise KeyError("Unsupport element, it should be a video or image, please check your input")
            query += text
            self.template.append_message(message.get("role"), query)

        prompt = self.template.get_prompt()
        self.template.messages.clear()

        query_ids = torch.tensor(self.tokenizer.encode(prompt))

        bos_pos_set = torch.nonzero(query_ids == self.img_begin_id).view(-1)
        eos_pos_set = torch.nonzero(query_ids == self.img_end_id).view(-1)
        for i, (bos_pos, eos_pos) in enumerate(zip(bos_pos_set, eos_pos_set)):
            if eos_pos - bos_pos < 3:
                logger.error("tokenize input error.",
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError("tokenize input error.")
            query_ids[bos_pos + 1] = shm_name_list[i]
            query_ids[bos_pos + 2] = shape_value_list[i]
            
        return query_ids
        

class InternVLSeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = SeparatorStyle.ADD_COLON_SINGLE
    ADD_COLON_TWO = SeparatorStyle.ADD_COLON_TWO
    ADD_COLON_SPACE_SINGLE = SeparatorStyle.ADD_COLON_SPACE_SINGLE
    NO_COLON_SINGLE = SeparatorStyle.NO_COLON_SINGLE
    NO_COLON_TWO = SeparatorStyle.NO_COLON_TWO
    ADD_NEW_LINE_SINGLE = SeparatorStyle.ADD_NEW_LINE_SINGLE
    LLAMA2 = SeparatorStyle.LLAMA2
    CHATGLM = SeparatorStyle.CHATGLM
    CHATML = SeparatorStyle.CHATML
    CHATINTERN = SeparatorStyle.CHATINTERN
    DOLLY = SeparatorStyle.DOLLY
    RWKV = SeparatorStyle.RWKV
    PHOENIX = SeparatorStyle.PHOENIX
    ROBIN = SeparatorStyle.ROBIN
    FALCON_CHAT = SeparatorStyle.FALCON_CHAT
    CHATGLM3 = SeparatorStyle.CHATGLM3
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclass
class InternVLConversation(Conversation):
    
    def get_prompt(self) -> str:
        """
            Get the prompt for generation.Different templates will be applied 
            based on the LLM separator style.
        """
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == InternVLSeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        return super.get_prompt()


def get_internvl_conv_template(name: str):
    if name not in conv_templates:
        logger.error(f'get conversation faild: cannot find the {name} template.',
                     ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
        raise ValueError(f'get conversation faild: cannot find the {name} template.')
    return conv_templates.get(name)


register_conv_template(
    InternVLConversation(
        name='Hermes-2',
        system_template='<|im_start|>system\n{system_message}',
        system_message=(
            '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
        ),
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        messages=[],
        sep_style=InternVLSeparatorStyle.MPT,
        sep='<|im_end|>',
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],
        stop_str='<|endoftext|>',
    )
)


register_conv_template(
    InternVLConversation(
        name='internlm2-chat',
        system_template='<|im_start|>system\n{system_message}',
        system_message=(
            '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
        ),
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        messages=[],
        sep_style=InternVLSeparatorStyle.MPT,
        sep='<|im_end|>',
        stop_token_ids=[
            2,  # </s>
            92543,  # <|im_start|>
            92542  # <|im_end|>
        ]
    )
)


register_conv_template(
    InternVLConversation(
        name='phi3-chat',
        system_template='<|system|>\n{system_message}',
        system_message=(
            '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
        ),
        roles=('<|user|>\n', '<|assistant|>\n'),
        messages=[],
        sep_style=InternVLSeparatorStyle.MPT,
        sep='<|end|>',
        stop_token_ids=[
            2,
            32000,
            32007
        ]
    )
)


register_conv_template(
    InternVLConversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message=(
            '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、'
            '清华大学及多家合作单位联合开发的多模态大语言模型。'
        ),
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=InternVLSeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)
