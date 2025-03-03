# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
from typing import List
from dataclasses import dataclass

import torch

from atb_llm.utils import file_utils


class Request:
    req_id: int

    input_ids: torch.Tensor
    input_length: int

    need_blocks: int
    need_slots: int
    block_tables: torch.Tensor
    slot_tables: torch.Tensor

    out_token_list: List[int]

    def __init__(self, max_out_length: int, block_size: int, req_id: int,
                 input_ids: torch.Tensor, adapter_id: None | str):
        self.req_id = req_id
        self.input_ids = input_ids.flatten()
        self.adapter_id = adapter_id

        self.input_length = self.input_ids.numel()

        try:
            self.need_blocks = math.ceil((self.input_length + max_out_length) / block_size)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        self.need_slots = self.need_blocks * block_size
        self.block_tables: None | torch.Tensor = None
        self.slot_tables: None | torch.Tensor = None

        self.out_token_list = []


class MultiModalRequest():
    def __init__(self, max_out_length: int, block_size: int, req_id: int,
                 input_ids: torch.Tensor, adapter_id: None | str, position_ids=None):
        self.req_id = req_id
        self.input_ids = input_ids
        self.adapter_id = adapter_id
        self.input_length = self.input_ids.shape[0]
        self.adapter_id = adapter_id
        self.position_ids = position_ids
        self.context_length = None if position_ids is None else position_ids[-1] + 1
        try:
            self.need_blocks = math.ceil((self.input_length + max_out_length) / block_size)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        self.need_slots = self.need_blocks * block_size
        self.block_tables = None
        self.slot_tables = None
        self.out_token_list = []


def request_from_token(input_ids, max_out_length, block_size,
                       req_idx=0, adapter_id=None) -> Request:
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
    request = Request(max_out_length, block_size, req_idx, input_ids, adapter_id)
    return request


def request_from_text(text, tokenizer, max_out_length, block_size, req_idx=0) -> Request:
    input_ids = tokenizer([text], return_tensors="pt")["input_ids"].flatten()
    request = request_from_token(input_ids, max_out_length, block_size, req_idx)
    return request


@dataclass
class MultiModalRequestParams:
    text:str
    image:str
    video:str
    max_out_length:int
    block_size:int
    req_idx:int
    adapter_id:str = None
    batch_size:int = 1
    

@dataclass
class MultiModalReqParams:
    text:List
    image:List
    video:List
    audio:List
    max_out_length:int
    block_size:int
    req_idx:int
    adapter_id:str = None
    batch_size:int = 1


def request_from_text_and_image(processor, model, multimodalparams):
    text = multimodalparams.text
    image = multimodalparams.image
    video = multimodalparams.video
    max_out_length = multimodalparams.max_out_length
    block_size = multimodalparams.block_size
    req_idx = multimodalparams.req_idx
    adapter_id = multimodalparams.adapter_id
    batch_size = multimodalparams.batch_size

    inputs_embeds = None
    position_ids = None
    prefill_inputs = model.model.prepare_prefill_token(text, image, video, processor, batch_size)
    if isinstance(prefill_inputs, tuple):
        inputs_embeds, position_ids = prefill_inputs
    else:
        inputs_embeds = prefill_inputs
    request = MultiModalRequest(max_out_length, block_size, req_idx, inputs_embeds, adapter_id, position_ids)
    return request


def request_from_multimodalinputs(processor, model, multimodalparams):
    max_out_length = multimodalparams.max_out_length
    block_size = multimodalparams.block_size
    req_idx = multimodalparams.req_idx
    adapter_id = multimodalparams.adapter_id
    
    inputs_embeds = None
    position_ids = None
    prefill_inputs = model.model.prepare_prefill_token(multimodalparams, processor)
    if isinstance(prefill_inputs, tuple):
        inputs_embeds, position_ids = prefill_inputs
    else:
        inputs_embeds = prefill_inputs
    request = MultiModalRequest(max_out_length, block_size, req_idx, inputs_embeds, adapter_id, position_ids)
    return request


def request_from_token_file(input_path, max_out_length, block_size) -> List[Request]:
    req_list = []
    req_idx = 0
    with file_utils.safe_open(input_path, 'r') as f:
        for line in file_utils.safe_readlines(f):
            token_str_list = line.split(',')
            input_ids = []
            for token_str in token_str_list:
                input_ids.append(int(token_str))
            req_list.append(request_from_token(input_ids, max_out_length, block_size, req_idx))
            req_idx += 1
    return req_list


def request_from_text_file(input_path, tokenizer, max_out_length, block_size) -> List[Request]:
    req_list = []
    req_idx = 0
    with file_utils.safe_open(input_path, 'r') as f:
        for line in file_utils.safe_readlines(f):
            if line[-1] != '\n':
                continue
            text = line[:-1]
            req_list.append(request_from_text(text, tokenizer, max_out_length, block_size, req_idx=0))
            req_idx += 1
    return req_list
