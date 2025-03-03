# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Part of this file was copied from project text-generation-inference 0.9.1
import json
import math
import os
from typing import List, Dict, Optional, Tuple

import torch
from safetensors import safe_open, SafetensorError

from .hub import weight_files
from .log import logger, print_log
from .quantize.quant_type import QuantType, LinearTypeV2, QUANTIZE_DESC_REQUIRED_LIST
from . import file_utils

QUANTIZE_DTYPE_LIST = [torch.int8, torch.int32, torch.int64]


class Weights:
    def __init__(
            self,
            model_name_or_path,
            device,
            dtype,
            process_group,
            quantize=None,
            extension: Optional[str] = ".safetensors",
            aliases: Optional[Dict[str, List[str]]] = None,
            **kwargs
    ):
        if quantize == QuantType.W8A8SC:
            model_name_or_path = os.path.join(model_name_or_path,
                                              f'part{process_group.rank()}-of-{process_group.size()}'
            )
            model_name_or_path = file_utils.standardize_path(model_name_or_path, check_link=False)
            file_utils.check_path_permission(model_name_or_path)
        self.filenames = weight_files(model_name_or_path, extension=extension)
        self.quantize = quantize
        routing = self.load_routing(process_group)
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self._handles = {}
        self.gptq_bits = None
        self.gptq_groupsize = None
        self.quant_desc = None

        self.init_quant_params(quantize, model_name_or_path)

    def release_file_handler(self):
        del self._handles
        self._handles = {}

    def load_routing(self, process_group):
        routing = {}
        for filename in self.filenames:
            filename = file_utils.standardize_path(str(filename), check_link=False)
            file_utils.check_path_permission(filename)
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        print_log(
                            process_group.rank(),
                            logger.error,
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}",
                            need_filter=True
                        )
                        raise AssertionError
                    routing[k] = filename
        return routing

    def get_linear_quant_type(self, key):
        if self.quant_desc is None:
            return LinearTypeV2.FLOAT16 if self.dtype == torch.float16 else LinearTypeV2.BFLOAT16
        if self.quant_desc.get(key, LinearTypeV2.INVALID) == "FLOAT":
            return LinearTypeV2.FLOAT16 if self.dtype == torch.float16 else LinearTypeV2.BFLOAT16
        return LinearTypeV2[self.quant_desc.get(key, LinearTypeV2.INVALID)]

    def correct_tensor_dtype(self, tensor, tensor_name):
        if tensor_name.endswith("deq_scale") and self.dtype == torch.bfloat16:
            # BF16场景下deq_scale字段的值为FP32
            return tensor
        if tensor.dtype not in [torch.int8, torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def init_quant_params(self, quantize, model_name_or_path):
        if quantize in QUANTIZE_DESC_REQUIRED_LIST:
            self._set_quant_params(model_name_or_path)

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
            raise AssertionError(f"weight {tensor_name} does not exist")
        return str(filename), tensor_name

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_whole_tensor(self, tensor_name: str, dim: int):
        slice_ = self._get_slice(tensor_name)

        start = 0
        stop = slice_.get_shape()[dim]

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            logger.error("Let's make that generic when needed")
            raise AssertionError
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_partial_sharded_mid_dim(self, tensor_name: str, dim: int, index: int = 1, gqa_size: int = 1):

        world_size = self.process_group.size()
        rank = self.process_group.rank()

        slice_ = self._get_slice(tensor_name)
        size = slice_.get_shape()[dim]

        block_size = size // 16
        start = block_size * index + rank * block_size // world_size
        stop = block_size * index + (rank + 1) * block_size // world_size

        if dim == 0:
            tensor = slice_[start:stop, :]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            logger.error("Let's make that generic when needed")
            raise AssertionError
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_partial_sharded(self, tensor_name: str, dim: int, gqa_size: int = 1):
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        slice_ = self._get_slice(tensor_name)
        size = slice_.get_shape()[dim]
        group_size = size // gqa_size
        if group_size >= world_size:
            block_size = size // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
        else:
            block_size = gqa_size
            start = (rank // (world_size // group_size)) * block_size
            stop = ((rank // (world_size // group_size)) + 1) * block_size

        if "c_attn.bias" in tensor_name:
            b = slice_[:]
            single_size = b.shape[0] // 3
            head_size = 128
            head_num = single_size // head_size
            rank_heads = math.ceil(head_num / world_size)
            if rank != world_size - 1:
                start = rank * (rank_heads * head_size)
                stop = (rank + 1) * (rank_heads * head_size)
                bq = slice_[start:stop]
                bk = slice_[start + single_size:stop + single_size]
                bv = slice_[start + 2 * single_size:stop + 2 * single_size]
            else:
                # last rank
                start = rank * (rank_heads * head_size)
                stop = head_num * head_size
                bq = slice_[start:stop]
                bk = slice_[start + single_size:stop + single_size]
                bv = slice_[start + 2 * single_size:stop + 2 * single_size]
            b_ = torch.cat([bq, bk, bv], dim=0)
            return b_

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            logger.error("Let's make that generic when needed")
            raise AssertionError
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_partial_sharded_padding(self, tensor_name: str, dim: int, gqa_size=1):
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        slice_ = self._get_slice(tensor_name)
        size = slice_.get_shape()[dim]

        head_num = size // gqa_size
        block_head_num = (head_num + world_size - 1) // world_size

        block_size = block_head_num * gqa_size

        start = rank * block_size
        stop = (rank + 1) * block_size

        if rank != world_size - 1:
            if dim == 0:
                tensor = slice_[start:stop]
            elif dim == 1:
                tensor = slice_[:, start:stop]
            else:
                logger.error("Let's make that generic when needed")
                raise AssertionError
        else:
            if dim == 0:
                tensor = slice_[start:]
            elif dim == 1:
                tensor = slice_[:, start:]
            else:
                logger.error("Let's make that generic when needed")
                raise AssertionError
        
        if len(tensor.shape) == 1:
            tensor_zeros = torch.zeros(size=(block_size,), dtype=tensor.dtype, device=tensor.device)
            tensor_zeros[:tensor.shape[0]] = tensor
            tensor = tensor_zeros
        else:
            dim0, dim1 = tensor.shape
            if dim == 0:
                dim0 = block_size
            else:
                dim1 = block_size
            tensor_zeros = torch.zeros(size=(dim0, dim1), dtype=tensor.dtype, device=tensor.device)
            tensor_zeros[:tensor.shape[0], :tensor.shape[1]] = tensor
            tensor = tensor_zeros

        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_sharded(self, tensor_name: str, dim: int, gqa_size: int = 1):
        slice_ = self._get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        if (size // gqa_size) % world_size == 0 or world_size % (size // gqa_size) == 0:
            return self.get_partial_sharded(tensor_name, dim, gqa_size)
        else:
            return self.get_partial_sharded_padding(tensor_name, dim, gqa_size)

    def get_per_tensor_sharded(self, prefixes, dim, tensor_name):
        tensor = torch.cat(
            [self.get_whole_tensor(f"{p}.{tensor_name}", dim=0) for p in prefixes], dim=dim
        )
        if torch.allclose(tensor, tensor[0]):
            tensor = tensor[:1]
        else:
            raise ValueError(f"`{tensor_name}` are not equal: {tensor}")
        return tensor
    
    def get_tensor_col_packed_qkv_mha(self, tensor_name: str, head_size: int = None, dim=0):
        slice_ = self._get_slice(tensor_name)
        total_size = slice_.get_shape()[-1 if dim == 1 else 0]
        if total_size % 3 != 0:
            raise ValueError("Prepacked qkv is not divisible by 3")
        single_size = total_size // 3
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        if dim == 1:
            if head_size is None:
                if single_size % world_size != 0:
                    raise RuntimeError(f"Prepacked qkv cannot be sharded across {world_size} shards")
                try:
                    block_size = single_size // world_size
                except ZeroDivisionError as e:
                    raise ZeroDivisionError from e
                start = rank * block_size
                stop = (rank + 1) * block_size
                if len(slice_.get_shape()) <= 1:
                    q = slice_[start:stop]
                    k = slice_[start + single_size:stop + single_size]
                    v = slice_[start + 2 * single_size:stop + 2 * single_size]
                    tensor = torch.cat([q, k, v], dim=0)
                else:
                    q = slice_[:, start:stop]
                    k = slice_[:, start + single_size:stop + single_size]
                    v = slice_[:, start + 2 * single_size:stop + 2 * single_size]
                    tensor = torch.cat([q, k, v], dim=1)
            else:
                raise ValueError("qkv are not supported")
        else:
            if head_size is None:
                if single_size % world_size != 0:
                    raise RuntimeError(f"Prepacked qkv cannot be sharded across {world_size} shards")
                try:
                    block_size = single_size // world_size
                except ZeroDivisionError as e:
                    raise ZeroDivisionError from e
                start = rank * block_size
                stop = (rank + 1) * block_size
                q = slice_[start:stop]
                k = slice_[start + single_size:stop + single_size]
                v = slice_[start + 2 * single_size:stop + 2 * single_size]
                tensor = torch.cat([q, k, v], dim=0)
            else:
                try:
                    head_num = single_size // head_size
                    rank_heads = math.ceil(head_num / world_size)
                except ZeroDivisionError as e:
                    raise ZeroDivisionError from e
                if rank != world_size - 1:
                    start = rank * (rank_heads * head_size)
                    stop = (rank + 1) * (rank_heads * head_size)
                    q = slice_[start:stop]
                    k = slice_[start + single_size:stop + single_size]
                    v = slice_[start + 2 * single_size:stop + 2 * single_size]
                    tensor = torch.cat([q, k, v], dim=0)
                else:
                    # last rank
                    start = rank * (rank_heads * head_size)
                    stop = head_num * head_size
                    q = slice_[start:stop]
                    k = slice_[start + single_size:stop + single_size]
                    v = slice_[start + 2 * single_size:stop + 2 * single_size]

                    # padding
                    q_zero = torch.zeros(size=(rank_heads * head_size, slice_.get_shape()[1]))
                    k_zero = torch.zeros(size=(rank_heads * head_size, slice_.get_shape()[1]))
                    v_zero = torch.zeros(size=(rank_heads * head_size, slice_.get_shape()[1]))
                    q_zero[:q.shape[0], :q.shape[1]] = q
                    k_zero[:k.shape[0], :k.shape[1]] = k
                    v_zero[:v.shape[0], :v.shape[1]] = v
                    tensor = torch.cat([q_zero, k_zero, v_zero], dim=0)
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_tensor_col_packed_o_gqa(self, tensor_name: str, hidden_size, num_heads, num_kv_heads):
        num_o_heads = num_heads
        head_size = hidden_size // num_heads

        slice_ = self.get_tensor(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        num_heads = math.ceil(num_heads / world_size)
        odd_rank_hidden_size = head_size * (num_heads - 1)
        even_rank_hidden_size = head_size * num_heads
        shape = list(slice_.shape)
        shape[0] = head_size
        group_rank = world_size // (num_heads * world_size - num_o_heads)
        padding_zero = torch.zeros(shape, dtype=slice_.dtype, device=slice_.device)
        if rank % group_rank == 0:
            start = (rank // group_rank) * ((group_rank - 1) * even_rank_hidden_size + odd_rank_hidden_size)
            indices = torch.range(start, start + odd_rank_hidden_size - 1).to(torch.int32)
            part_tensor = torch.index_select(slice_, 1, indices)
            part_tensor = torch.cat((part_tensor, padding_zero.T), dim=1)
        else:
            start = (rank // group_rank) * ((group_rank - 1) * even_rank_hidden_size + odd_rank_hidden_size) + (
                        (rank % group_rank - 1) * even_rank_hidden_size + odd_rank_hidden_size)
            start = int(start)
            indices = torch.range(start, start + even_rank_hidden_size - 1).to(torch.int32)
            part_tensor = torch.index_select(slice_, 1, indices)
        return part_tensor

    def get_tensor_col_packed_q_gqa(self, tensor_name: str, hidden_size, num_heads, num_kv_heads):
        num_q_heads = num_heads
        head_size = hidden_size // num_heads

        slice_ = self.get_tensor(tensor_name)
        world_size = self.process_group.size()
        num_heads = math.ceil(num_heads / world_size)
        rank = self.process_group.rank()
        odd_rank_hidden_size = head_size * (num_heads - 1)
        even_rank_hidden_size = head_size * num_heads
        shape = list(slice_.shape)
        shape[0] = head_size
        group_rank = world_size // (num_heads * world_size - num_q_heads)
        padding_zero = torch.zeros(shape, dtype=slice_.dtype, device=slice_.device)
        if rank % group_rank == 0:
            start = (rank // group_rank) * ((group_rank - 1) * even_rank_hidden_size + odd_rank_hidden_size)
            indices = torch.range(start, start + odd_rank_hidden_size - 1).to(torch.int32)
            part_tensor = torch.index_select(slice_, 0, indices)
            part_tensor = torch.cat((part_tensor, padding_zero), dim=0)
        else:
            start = (rank // group_rank) * ((group_rank - 1) * even_rank_hidden_size + odd_rank_hidden_size) + (
                        (rank % group_rank - 1) * even_rank_hidden_size + odd_rank_hidden_size)
            start = int(start)
            indices = torch.range(start, start + even_rank_hidden_size - 1).to(torch.int32)
            part_tensor = torch.index_select(slice_, 0, indices)
        return part_tensor

    def get_tensor_col_packed_k_gqa(self, tensor_name: str, hidden_size, num_heads, num_kv_heads):
        slice_ = self.get_tensor(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        kv_tp_size = min(world_size, num_kv_heads)
        key_list = torch.chunk(slice_, kv_tp_size, dim=0)
        tensor = key_list[rank * kv_tp_size // world_size]
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_tensor_col_packed_v_gqa(self, tensor_name: str, hidden_size, num_heads, num_kv_heads):
        slice_ = self.get_tensor(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        kv_tp_size = min(world_size, num_kv_heads)
        value_list = torch.chunk(slice_, kv_tp_size, dim=0)
        tensor = value_list[rank * kv_tp_size // world_size]
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_tensor_col_packed_qkv_gqa(self, tensor_name: str, num_heads, num_kv_heads):
        slice_ = self.get_tensor(tensor_name)
        total_size = slice_.shape[0]
        if total_size % (num_heads + num_kv_heads * 2) != 0:
            raise AssertionError("Prepacked qkv is not divisible by q,k,v")
        q_single_size = total_size * num_heads // (num_heads + num_kv_heads * 2)
        kv_single_size = total_size * num_kv_heads // (num_heads + num_kv_heads * 2)
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        if q_single_size % world_size != 0:
            raise AssertionError(f"Prepacked qkv cannot be sharded across {world_size} shards")
        query_layer, key_layer, value_layer = slice_.split((q_single_size, kv_single_size, kv_single_size), dim=0)
        kv_tp_size = min(world_size, num_kv_heads)
        query_list = torch.chunk(query_layer, world_size, dim=0)
        key_list = torch.chunk(key_layer, kv_tp_size, dim=0)
        value_list = torch.chunk(value_layer, kv_tp_size, dim=0)
        tensor = torch.cat([query_list[rank],
                            key_list[rank * kv_tp_size // world_size],
                            value_list[rank * kv_tp_size // world_size]], dim=0)
        return self.correct_tensor_dtype(tensor, tensor_name)
    
    def get_tensor_col_packed_kv_mha(self, tensor_name: str, hiden_size, head_size: int = None):
        slice_ = self._get_slice(tensor_name)
        total_size = slice_.get_shape()[0]
        if total_size % 2 != 0:
            raise ValueError("Prepacked qkv is not divisible by 2")
        single_size = total_size // 2
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        if head_size is None:
            raise RuntimeError("head_size is neccessary")
        else:
            try:
                head_num = single_size // head_size
                rank_heads = math.ceil(head_num / world_size)
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e
        
            start = rank * (rank_heads * head_size * 2)
            stop = (rank + 1) * (rank_heads * head_size * 2)
            kv = slice_[start:stop]
            kv_new = kv.reshape(rank_heads, 2, head_size, -1)
            k, v = torch.chunk(kv_new, 2, dim=1)
            if len(slice_.get_shape()) == 1:
                k = k.reshape(head_size * rank_heads)
                v = v.reshape(head_size * rank_heads)
            else:
                k = k.reshape(head_size * rank_heads, hiden_size)
                v = v.reshape(head_size * rank_heads, hiden_size)
            tensor = torch.cat([k, v], dim=0)
        
        return self.correct_tensor_dtype(tensor, tensor_name)


    def get_tensor_col_packed_o(self, tensor_name: str, hidden_size, num_heads, num_kv_heads=None):
        return self.get_tensor_col_packed_o_gqa(tensor_name, hidden_size, num_heads, num_kv_heads)

    def get_tensor_col_packed_q(self, tensor_name: str, hidden_size, num_heads, num_kv_heads=None):
        return self.get_tensor_col_packed_q_gqa(tensor_name, hidden_size, num_heads, num_kv_heads)

    def get_tensor_col_packed_k(self, tensor_name: str, hidden_size, num_heads, num_kv_heads=None):
        return self.get_tensor_col_packed_k_gqa(tensor_name, hidden_size, num_heads, num_kv_heads)

    def get_tensor_col_packed_v(self, tensor_name: str, hidden_size, num_heads, num_kv_heads=None):
        return self.get_tensor_col_packed_v_gqa(tensor_name, hidden_size, num_heads, num_kv_heads)

    def get_w8a8sc_weight(self, prefix: str):
        qweight = self.get_tensor(f"{prefix}.weight")
        if qweight.dtype in [torch.float16, torch.bfloat16]:
            return qweight
        deq_scale = self.get_tensor(f"{prefix}.deq_scale")
        quant_bias = self.get_tensor(f"{prefix}.quant_bias")
        input_scale = self.get_tensor(f"{prefix}.input_scale")
        input_offset = self.get_tensor(f"{prefix}.input_offset")
        index = self.get_tensor(f"{prefix}.index")
        weight = (qweight, deq_scale, quant_bias, input_scale, input_offset, index)
        return weight

    def get_weights_col_packed_kv(self, prefix: str, quantize: str, hidden_size, head_size, num_kv_heads=None):
        if quantize in [QuantType.W8A8, QuantType.W8A8S]:
            qweight = self.get_tensor_col_packed_kv_mha(f"{prefix}.weight", hidden_size, head_size)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = self.get_tensor_col_packed_kv_mha(f"{prefix}.deq_scale", hidden_size, head_size)
            quant_bias = self.get_tensor_col_packed_kv_mha(f"{prefix}.quant_bias", hidden_size, head_size)
            input_scale = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_scale')
            input_offset = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_offset')
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset)
        elif quantize in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            qweight = self.get_tensor_col_packed_kv_mha(f"{prefix}.weight", hidden_size, head_size)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            weight_scale = self.get_tensor_col_packed_kv_mha(f"{prefix}.weight_scale", hidden_size, head_size)
            weight_offset = self.get_tensor_col_packed_kv_mha(f"{prefix}.weight_offset", hidden_size, head_size)
            weight = (qweight, weight_scale, weight_offset)
        elif quantize == QuantType.W8A8SC:
            return self.get_w8a8sc_weight(prefix)
        else:
            weight = self.get_tensor_col_packed_kv_mha(f"{prefix}.weight", hidden_size, head_size)
        return weight


    def get_weights_col_packed_o(self, prefix: str, quantize: str, hidden_size, num_heads, num_kv_heads=None):
        weight = self.get_tensor_col_packed_o(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads)
        return weight

    def get_weights_col_packed_q(self, prefix: str, quantize: str, hidden_size, num_heads, num_kv_heads=None):
        weight = self.get_tensor_col_packed_q(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads)
        return weight

    def get_weights_col_packed_k(self, prefix: str, quantize: str, hidden_size, num_heads, num_kv_heads=None):
        weight = self.get_tensor_col_packed_k(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads)
        return weight

    def get_weights_col_packed_v(self, prefix: str, quantize: str, hidden_size, num_heads, num_kv_heads=None):
        weight = self.get_tensor_col_packed_v(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads)
        return weight


    def get_tensor_col_packed_qkv(self, tensor_name: str, hidden_size, num_heads, num_kv_heads=None, dim=0):
        if not num_kv_heads:
            num_kv_heads = num_heads
        if num_heads == num_kv_heads:
            if num_heads % self.process_group.size() == 0:
                return self.get_tensor_col_packed_qkv_mha(tensor_name, dim=dim)
            else:
                return self.get_tensor_col_packed_qkv_mha(tensor_name, hidden_size // num_heads, dim=dim)
        else:
            return self.get_tensor_col_packed_qkv_gqa(tensor_name, num_heads, num_kv_heads)

    def get_weights_col_packed_qkv(self, prefix: str, quantize: str, hidden_size, num_heads, num_kv_heads=None, dim=0):
        if quantize in [QuantType.W8A8, QuantType.W8A8S]:
            qweight = self.get_tensor_col_packed_qkv(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = self.get_tensor_col_packed_qkv(f"{prefix}.deq_scale", hidden_size, num_heads, num_kv_heads)
            quant_bias = self.get_tensor_col_packed_qkv(f"{prefix}.quant_bias", hidden_size, num_heads, num_kv_heads)
            input_scale = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_scale')
            input_offset = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_offset')
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset)
        elif quantize in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            qweight = self.get_tensor_col_packed_qkv(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            weight_scale = self.get_tensor_col_packed_qkv(f"{prefix}.weight_scale", hidden_size, num_heads,
                                                          num_kv_heads)
            weight_offset = self.get_tensor_col_packed_qkv(f"{prefix}.weight_offset", hidden_size, num_heads,
                                                           num_kv_heads)
            weight = (qweight, weight_scale, weight_offset)
        elif quantize == QuantType.W8A8SC:
            return self.get_w8a8sc_weight(prefix)
        else:
            weight = self.get_tensor_col_packed_qkv(f"{prefix}.weight", hidden_size, num_heads, num_kv_heads, dim=dim)
        return weight

    def get_tensor_col_packed_mlp(self, tensor_name, head_types=2):
        slice_ = self.get_tensor(tensor_name)
        total_size = slice_.shape[0]
        if total_size % head_types != 0:
            raise AssertionError("Prepacked mlp is not divisible by up,gate")
        up_single_size = total_size // head_types
        gate_single_size = total_size // head_types
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        if up_single_size % world_size != 0:
            raise AssertionError(f"Prepacked mlp cannot be sharded across {world_size} shards")
        gate_layer, up_layer = slice_.split((up_single_size, gate_single_size), dim=0)
        gate_list = torch.chunk(gate_layer, world_size, dim=0)
        up_list = torch.chunk(up_layer, world_size, dim=0)
        tensor = torch.cat([gate_list[rank], up_list[rank]], dim=0)
        return self.correct_tensor_dtype(tensor, tensor_name)

    def get_weights_col_packed_mlp(self, prefix: str, quantize: str):
        if quantize in [QuantType.W8A8, QuantType.W8A8S]:
            qweight = self.get_tensor_col_packed_mlp(f"{prefix}.weight")
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = self.get_tensor_col_packed_mlp(f"{prefix}.deq_scale")
            quant_bias = self.get_tensor_col_packed_mlp(f"{prefix}.quant_bias")
            input_scale = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_scale')
            input_offset = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_offset')
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset)
        elif quantize in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            qweight = self.get_tensor_col_packed_mlp(f"{prefix}.weight")
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            weight_scale = self.get_tensor_col_packed_mlp(f"{prefix}.weight_scale")
            weight_offset = self.get_tensor_col_packed_mlp(f"{prefix}.weight_offset")
            weight = (qweight, weight_scale, weight_offset)
        elif quantize == QuantType.W8A8SC:
            return self.get_w8a8sc_weight(prefix)
        else:
            weight = self.get_tensor_col_packed_mlp(f"{prefix}.weight")
        return weight

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int, gqa_size: int = 1):
        if quantize == "gptq":
            try:
                qweight = torch.cat(
                    [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError as err:
                logger.error(
                    "Cannot load `gptq` weight, make sure the model is already quantized"
                )
                raise AssertionError from err

            qzeros = torch.cat(
                [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
            )
            scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
            )
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        elif quantize in [QuantType.W8A8, QuantType.W8A8S]:
            qweight = torch.cat(
                [self.get_sharded(f"{p}.weight", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = torch.cat(
                [self.get_sharded(f"{p}.deq_scale", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            quant_bias = torch.cat(
                [self.get_sharded(f"{p}.quant_bias", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            input_scale = self.get_per_tensor_sharded(prefixes, dim, 'input_scale')
            input_offset = self.get_per_tensor_sharded(prefixes, dim, 'input_offset')
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset)
        elif quantize in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            qweight = torch.cat(
                [self.get_sharded(f"{p}.weight", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            weight_scale = torch.cat(
                [self.get_sharded(f"{p}.weight_scale", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            weight_offset = torch.cat(
                [self.get_sharded(f"{p}.weight_offset", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            weight = (qweight, weight_scale, weight_offset)
        elif quantize == QuantType.W8A8SC:
            qweight = torch.cat([self.get_tensor(f"{p}.weight") for p in prefixes], dim=dim)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = torch.cat([self.get_tensor(f"{p}.deq_scale") for p in prefixes], dim=dim)
            quant_bias = torch.cat([self.get_tensor(f"{p}.quant_bias") for p in prefixes], dim=dim)
            input_scale = torch.cat([self.get_tensor(f"{p}.input_scale") for p in prefixes], dim=dim)
            input_offset = torch.cat([self.get_tensor(f"{p}.input_offset") for p in prefixes], dim=dim)
            index = torch.cat([self.get_tensor(f"{p}.index") for p in prefixes], dim=dim)
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset, index)
        else:
            w = [self.get_sharded(f"{p}.weight", dim=dim, gqa_size=gqa_size) for p in prefixes]
            weight = torch.cat(w, dim=dim)
        return weight

    def get_multi_weights_row(self, prefix: str, quantize: str, gqa_size=1, dim=1):
        if quantize in [QuantType.W8A8, QuantType.W8A8S]:
            qweight = self.get_sharded(f"{prefix}.weight", dim=dim, gqa_size=gqa_size)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = self.get_tensor(f"{prefix}.deq_scale")
            quant_bias = self.get_tensor(f"{prefix}.quant_bias")
            if self.process_group.rank() == 0:
                quant_bias = quant_bias
            else:
                quant_bias = torch.zeros_like(quant_bias, dtype=quant_bias.dtype, device=quant_bias.device)
            input_scale = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_scale')
            input_offset = self.get_per_tensor_sharded([prefix], dim=0, tensor_name='input_offset')
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset)
        elif quantize in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            qweight = self.get_sharded(f"{prefix}.weight", dim=1, gqa_size=gqa_size)
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            weight_scale = self.get_sharded(f"{prefix}.weight_scale", dim=dim, gqa_size=1)
            weight_offset = self.get_sharded(f"{prefix}.weight_offset", dim=dim, gqa_size=1)
            weight = (qweight, weight_scale, weight_offset)
        elif quantize == QuantType.W8A8SC:
            return self.get_w8a8sc_weight(prefix)
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=dim, gqa_size=gqa_size)
        return weight

    def get_replicated_weights(self, prefix: str, quantize: str):
        if quantize in [QuantType.W8A8, QuantType.W8A8S]:
            qweight = self.get_tensor(f"{prefix}.weight")
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            deq_scale = self.get_tensor(f"{prefix}.deq_scale")
            quant_bias = self.get_tensor(f"{prefix}.quant_bias")
            input_scale = self.get_tensor(f"{prefix}.input_scale")
            input_offset = self.get_tensor(f"{prefix}.input_offset")
            weight = (qweight, deq_scale, quant_bias, input_scale, input_offset)
        elif quantize in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            qweight = self.get_tensor(f"{prefix}.weight")
            if qweight.dtype in [torch.float16, torch.bfloat16]:
                return qweight
            weight_scale = self.get_tensor(f"{prefix}.weight_scale")
            weight_offset = self.get_tensor(f"{prefix}.weight_offset")
            weight = (qweight, weight_scale, weight_offset)
        elif quantize == QuantType.W8A8SC:
            return self.get_w8a8sc_weight(prefix)
        else:
            weight = self.get_tensor(f"{prefix}.weight")
        return weight

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f
        return self._handles[filename]

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def _get_gptq_params(self) -> Tuple[int, int]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
        except (SafetensorError, RuntimeError) as _:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
            except Exception as err:
                raise AssertionError from err

        return bits, groupsize

    def _set_quant_params(self, model_id):
        try:
            filename = os.path.join(model_id, f'quant_model_description_{self.quantize}.json')
            with file_utils.safe_open(filename, 'r') as f:
                data = json.load(f)
            self.quant_desc = data
        except Exception as err:
            raise AssertionError from err
