# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestReshapeAndCacheOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "ReshapeAndCache"
        self.op_param = {}
        self.op_name = "ReshapeAndCacheOperation"
        self.op_set = (self.op_type, self.op_param, self.op_name)

        self.num_head = 32
        self.head_size = 128
        self.block_size = 128
        self.num_blocks = 512
        self.num_tokens = random.randint(2, 10)
        self.head_size_k = random.randint(1, 256)
        self.head_size_v = random.randint(1, 256)
        self.m = random.randint(2, 10)
        self.n = random.randint(2, 20)

    def generate_data(self, datatype=torch.float16):
        key = torch.rand(self.num_tokens, self.num_head,
                         self.head_size_k, dtype=datatype)
        value = torch.rand(self.num_tokens, self.num_head,
                           self.head_size_v, dtype=datatype)

        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(num_slots), self.num_tokens)
        slot_mapping = torch.tensor(slot_list, dtype=torch.int32)

        key_cache = torch.ones(size=(
            self.num_blocks, self.block_size, self.num_head, self.head_size_k), dtype=datatype)
        value_cache = torch.ones(size=(
            self.num_blocks, self.block_size, self.num_head, self.head_size_v), dtype=datatype)

        ret_data = key, value, key_cache, value_cache, slot_mapping
        return ret_data

    def get_golden(self, inputs):

        key, value, key_cache, value_cache, slot_mapping = \
            inputs['in0'], inputs['in1'], inputs['in2'], inputs['in3'], inputs['in4']
        golden_key_cache = torch.ones_like(key_cache, dtype=key_cache.dtype)
        golden_value_cache = torch.ones_like(
            value_cache, dtype=value_cache.dtype)

        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size
            golden_key_cache[block_index][block_offset] = key[i]
            golden_value_cache[block_index][block_offset] = value[i]

        return [golden_key_cache.npu(), golden_value_cache.npu()]

    def test_float16(self):
        key, value, key_cache, value_cache, slot_mapping = self.generate_data()

        inputs = {
            'in0': key.npu(), 
            'in1': value.npu(), 
            'in2': key_cache.npu(), 
            'in3': value_cache.npu(), 
            'in4': slot_mapping.npu()
        }
        outputs = {'out0': key_cache.npu(), 'out1': value_cache.npu()}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
