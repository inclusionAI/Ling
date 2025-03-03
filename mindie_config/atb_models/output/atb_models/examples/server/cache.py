# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import torch

from atb_llm.utils.log import logger
from atb_llm.utils.env import ENV


class CacheConfig:
    def __init__(self, num_blocks=1024, block_size=128, input_max_length=2048, output_max_length=128, batch_size=1,
                 rank=0, world_size=1):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size


class ModelConfig:
    def __init__(self, num_heads, num_kv_heads, num_kv_heads_origin, k_head_size, v_head_size,
                num_layers, device, dtype, soc_info, kv_quant_type, fa_quant_type=None):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_origin = num_kv_heads_origin
        self.head_size = k_head_size
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.soc_info = soc_info
        self.kv_quant_type = kv_quant_type
        self.fa_quant_type = fa_quant_type

    def __repr__(self):
        return (
                "ModelConfig("
                + f"num_heads={self.num_heads}, "
                + f"num_kv_heads={self.num_kv_heads}, "
                + f"num_kv_heads_origin={self.num_kv_heads_origin}, "
                + f"head_size={self.head_size}, "
                + f"k_head_size={self.k_head_size}, "
                + f"v_head_size={self.v_head_size}, "
                + f"num_layers={self.num_layers}, "
                + f"device={self.device}, "
                + f"dtype={self.dtype}, "
                + f"soc_info={self.soc_info}, "
                + f"kv_quant_type={self.kv_quant_type}, "
                + f"fa_quant_type={self.fa_quant_type}, "
        )


class CacheManager:
    def __init__(self, cache_config, model_config):
        self.block_size = cache_config.block_size
        self.num_blocks = cache_config.num_blocks
        self.new_num_blocks = self.num_blocks
        self.input_max_length = cache_config.input_max_length
        self.output_max_length = cache_config.output_max_length
        self.batch_size = cache_config.batch_size
        self.rank = cache_config.rank
        self.world_size = cache_config.world_size
        
        compress_head_enable = ENV.compress_head_enable
        compress_head_rope = ENV.compress_head_rope
        self.num_heads = 1 if compress_head_enable else model_config.num_kv_heads
        self.k_head_size = model_config.k_head_size
        self.v_head_size = model_config.v_head_size
        self.num_layers = model_config.num_layers
        self.device = model_config.device
        self.dtype = torch.int8 if model_config.kv_quant_type is not None or \
            model_config.fa_quant_type is not None else model_config.dtype
        self.soc_info = model_config.soc_info

        mem_need = self.num_blocks * self.block_size * self.num_heads * (self.k_head_size + self.v_head_size) * \
            self.num_layers * self.get_dtype_size(self.dtype) / 1024 / 1024 / 1024
        logger.info(f"kv cache will allocate {mem_need}GB memory")

        if compress_head_enable:
            if compress_head_rope:
                #仅针对llama3.1 70B
                if self.num_layers == 80:
                    #RA特性对于Rope位置编码的头部压缩字典
                    head_dict = {
                        'prefix_matching': {0:[0, 1, 2, 3, 4, 5, 6, 7], 1:[0, 1, 2, 3, 4, 5, 6, 7],
                                18: [5, 0, 2, 6, 1, 7], 76: [7, 1, 5, 3], 35: [5, 3, 2, 0, 6, 4],
                                74: [0, 3, 2, 1, 5], 52: [2, 7, 4, 6, 3], 56: [0, 5, 2, 3, 4],
                                77: [6, 0, 5, 1, 7, 4], 64: [3, 6], 33: [2, 1, 5, 4], 53: [0, 2],
                                37: [3, 4, 1, 2, 0, 5], 54: [1, 3], 21: [4, 6, 7, 3, 2], 47: [2],
                                72: [4], 31: [2, 4, 5, 3, 0, 6], 44: [5, 0], 67: [7, 6],
                                22: [1, 6, 4, 3], 68: [6, 2, 1, 7], 23: [5, 1, 4, 2, 6],
                                71: [0, 6, 4, 1], 39: [5, 1, 4], 36: [6, 5, 1, 2, 0],
                                27: [4, 2, 6, 5, 1], 73: [0, 4, 7, 5, 2, 1], 30: [6, 4, 5, 3],
                                14: [4, 2], 38: [6, 2, 3, 7], 60: [2, 3, 7, 6], 34: [0, 3, 4, 5],
                                41: [7, 2, 4], 19: [4, 0], 69: [4, 5], 29: [1, 6, 3, 7],
                                75: [2, 3, 0, 4, 5, 1], 61: [0, 4], 49: [0],
                                25: [1, 7, 4], 57: [0, 1, 3, 6], 17: [6, 3, 7, 1, 2], 58: [0],
                                24: [0, 5, 3], 32: [2, 5, 6], 42: [3], 55: [0], 70: [1, 6], 28: [5, 1, 0],
                                48: [7, 6], 50: [4], 16: [4, 5], 7: [3, 0], 63: [7, 4], 51: [3], 78: [0],
                                5: [5], 59: [4], 26: [3, 4, 5], 66: [0], 15: [2], 40: [4], 43: [2],
                                45: [0], 6: [4]},
                        'copying': {14: [2], 17: [7], 74: [0], 33: [2], 52: [2], 15: [2],
                                    31: [4], 38: [6], 71: [0], 23: [5], 27: [4, 2], 30: [4, 6],
                                    19: [0], 77: [5], 75: [3], 47: [2], 21: [6]}
                    }
                    inductive_head = head_dict["prefix_matching"]
                    copying_head = head_dict["copying"]
                    first_sink = 40
                    last_sink = max(4000, self.input_max_length // 5)
                    self.new_layers_num_blocks = []
                    kv_tp_size = min(cache_config.world_size, model_config.num_kv_heads_origin)
                    for layer_idx in range(self.num_layers):
                        global_need_block = 0
                        for head_idx in range(model_config.num_kv_heads):
                            cur_head_idx = head_idx + self.rank * kv_tp_size // \
                                self.world_size * model_config.num_kv_heads
                            is_inductive_head = layer_idx in inductive_head \
                                and cur_head_idx in inductive_head.get(layer_idx)
                            is_copying_head = layer_idx in copying_head and cur_head_idx in copying_head[layer_idx]
                            if (is_inductive_head or is_copying_head) or \
                                (self.input_max_length - first_sink - last_sink - 1 <= 0):
                                temp_length = self.input_max_length + self.output_max_length
                            else:
                                temp_length = first_sink + 1 + last_sink + self.output_max_length

                            need_block = math.ceil(temp_length / self.block_size)
                            global_need_block = global_need_block + need_block
                        self.new_layers_num_blocks.append(global_need_block)
                else:
                    self.new_layers_num_blocks = self.new_num_blocks * model_config.num_kv_heads
            else:
                wins = [
                    105, 125, 148, 176, 210, 250, 297, 353, 420, 500, 595, 707, 841, 1001, 1190, 1415, 1683, 2002, 2381,
                    2831, 3367, 4004, 4762, 5663, 6734, 8008, 9524, 11326, 13469, 16017, 19048, 22652
                ]
                if self.num_layers == 40:
                    wins = [
                        105, 125, 149, 178, 211, 251, 299, 356, 423, 503,
                        598, 712, 847, 1007, 1198, 1424, 1694, 2014, 2396, 2849,
                        3388, 4031, 4791, 5699, 6779, 8061, 9583, 11399, 13559, 16117,
                        19176, 22790, 97, 115, 137, 163, 194, 230, 274, 326
                    ]
                temp_c = self.input_max_length
                all_block_num = 0
                temp_length = 0
                num_block = 0
                for wins_item in enumerate(wins):
                    wins_index = wins_item[0]
                    wins_val = wins_item[1]
                    temp_length = min(wins_val, temp_c) + self.output_max_length
                    if self.block_size != 0:
                        num_block = num_block + math.ceil(temp_length / self.block_size)
                    if (wins_index + 1) % model_config.num_kv_heads_origin == 0:
                        all_block_num = max(all_block_num, num_block)
                        temp_length = 0
                        num_block = 0
                self.new_num_blocks = all_block_num * self.batch_size + 100


        if self.soc_info.need_nz:
            v_cache_shape = (self.new_num_blocks, self.num_heads * self.v_head_size // 16, self.block_size, 16)
            if self.v_head_size == 0:
                v_cache_shape = (1,)
            self.kv_cache = [
                (
                    torch.empty(
                        (self.new_num_blocks, self.num_heads * self.k_head_size // 16, self.block_size, 16),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        v_cache_shape,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                for _ in range(self.num_layers)
            ]
        else:
            if compress_head_rope:
                #仅针对llama3.1 70B
                if self.num_layers == 80:
                    self.kv_cache = [
                        (
                            torch.empty(
                                (self.new_layers_num_blocks[i], self.block_size, self.num_heads, self.k_head_size),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.empty(
                                (self.new_layers_num_blocks[i], self.block_size, self.num_heads, self.v_head_size),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        )
                        for i in range(self.num_layers)
                    ]
                else:
                    self.kv_cache = [
                        (
                            torch.empty(
                                (self.new_layers_num_blocks, self.block_size, self.num_heads, self.k_head_size),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.empty(
                                (self.new_layers_num_blocks, self.block_size, self.num_heads, self.v_head_size),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        )
                        for i in range(self.num_layers)
                    ]
            else:
                v_cache_shape = (self.new_num_blocks, self.block_size, self.num_heads, self.v_head_size)
                if self.v_head_size == 0:
                    v_cache_shape = (1,)
                self.kv_cache = [
                    (
                        torch.empty(
                            (self.new_num_blocks, self.block_size, self.num_heads, self.k_head_size),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        torch.empty(
                            v_cache_shape,
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    )
                    for _ in range(self.num_layers)
                ]

        random_block_allocate = False
        if random_block_allocate:
            self.block_map = torch.randperm(self.new_num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.zeros(self.new_num_blocks, dtype=torch.long)
            for i in range(self.new_num_blocks):
                self.contrary_block_map[self.block_map[i]] = i
        else:
            self.block_map = torch.arange(self.new_num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.arange(self.new_num_blocks, dtype=torch.long)

        self.free_block_mask = torch.ones(self.new_num_blocks, dtype=torch.long)
        self.total_slots = torch.arange(self.new_num_blocks * self.block_size, dtype=torch.long)
        self.total_slots = self.total_slots.view(self.new_num_blocks, self.block_size)

    @staticmethod
    def get_dtype_size(dtype):
        dtype_size_map = {torch.float16: 2, torch.float32: 4, torch.bfloat16: 2, torch.int8: 1}
        return dtype_size_map.get(dtype, 2)

    def allocate(self, batch):
        total_need_blocks = 0
        max_need_blocks = 0
        for req in batch.req_list:
            if req.block_tables:
                logger.error(f"req_id: {req.req_id} block has been allocated")
                raise AssertionError

            total_need_blocks += req.need_blocks
            max_need_blocks = max(max_need_blocks, req.need_blocks)

        free_block_indices = self.free_block_mask.nonzero().flatten()
        if free_block_indices.numel() < total_need_blocks:
            logger.error(f"Out of available cache blocks: asked {total_need_blocks}, "
                         f"only {free_block_indices.numel()} free blocks")
            raise AssertionError

        allocate_block_indices = free_block_indices[:total_need_blocks]
        allocate_blocks = self.block_map[allocate_block_indices]

        block_offset = 0
        block_tables_list = []
        slot_tables_list = []
        for req in batch.req_list:
            req.block_tables = allocate_blocks[block_offset:block_offset + req.need_blocks]
            req.slot_tables = self.total_slots[req.block_tables].flatten()
            block_tables = req.block_tables
            if req.need_blocks < max_need_blocks:
                block_tables = torch.concat(
                    [block_tables, torch.zeros(max_need_blocks - req.need_blocks, dtype=torch.long)], dim=0)
            block_tables_list.append(block_tables.view(1, -1))
            slot_tables_list.append(req.slot_tables)
            block_offset += req.need_blocks

        batch.batch_block_tables = torch.concat(block_tables_list, dim=0)
        batch.batch_slots_tables = torch.concat(slot_tables_list, dim=0)

        self.free_block_mask[allocate_block_indices] = 0

    def free(self, req):
        if req.block_tables is not None:
            block_indices = self.contrary_block_map[req.block_tables]
            self.free_block_mask[block_indices] = 1

    def get_free_block_num(self):
        free_block_indices = self.free_block_mask.nonzero()
        return len(free_block_indices)
