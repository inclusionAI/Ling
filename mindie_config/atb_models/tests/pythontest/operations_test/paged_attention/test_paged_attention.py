# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestPagedAttentionOperation(operation_test.OperationTest):
    def setUp(self):
        self.batch = random.randint(2, 8)
        self.head_num = random.randint(1, 4) * 2
        self.head_size = 128
        self.head_size_v = 16 * random.randint(1, 8)
        self.block_size = 128
        self.block_num = 64
        self.qk_scale = 1
        self.max_context_len = 1024
        self.max_block_nums_per_query = (self.max_context_len + self.block_size - 1) // self.block_size

        self.op_type = "PagedAttention"
        self.op_name = "PagedAttentionOperation"
        self.op_param = {
            "headNum": self.head_num,
            "qkScale": float(self.qk_scale),
        }
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensors):
        q, cache_k, cache_v, block_tables, context_lens = \
            in_tensors['in0'], in_tensors['in1'], in_tensors['in2'], in_tensors['in3'], in_tensors['in4']

        q = q.type(torch.float32)
        cache_k = cache_k.type(torch.float32)
        cache_v = cache_v.type(torch.float32)

        context = []
        for i, context_len in enumerate(context_lens):
            block_table = block_tables[i]
            block_num = context_len // self.block_size
            context_len = context_len % self.block_size
            blocks = block_table[:block_num]

            cur_q = q[i:i + 1]    # [q_len, head_num, head_size]
            cur_k = cache_k[blocks]  # [kv_len, head_num, head_size]
            cur_v = cache_v[blocks]  # [kv_len, head_num, head_size_v]

            cur_k = cur_k.view(-1, self.head_num, self.head_size)
            cur_v = cur_v.view(-1, self.head_num, self.head_size_v)

            if context_len != 0:
                cur_k = torch.concat((cur_k, cache_k[block_table[block_num], :context_len]), dim=0)
                cur_v = torch.concat((cur_v, cache_v[block_table[block_num], :context_len]), dim=0)

            cur_q = cur_q.permute(1, 0, 2)  # [head_num, q_len, head_size]
            cur_k = cur_k.permute(1, 2, 0)  # [head_num, head_size, kv_len]
            cur_qk = (torch.bmm(cur_q, cur_k) * self.qk_scale)  # [head_num, q_len, kv_len]
            cur_qk = torch.nn.functional.softmax(cur_qk, dim=-1)

            cur_v = cur_v.permute(1, 0, 2)  # [head_num, kv_len, head_size_v]
            cur_context = torch.bmm(cur_qk, cur_v)  # [head_num, q_len, head_size_v]
            cur_context = cur_context.permute(1, 0, 2).contiguous() # [q_len, head_num, head_size_v]

            context.append(cur_context)

        context = torch.concat(context, dim=0).type(torch.float16)
        return [context.npu()]

    def test_float16(self):
        seq_len = torch.ones((self.batch,), dtype=torch.int32)
        ntokens = int(seq_len.sum())

        q = torch.rand(ntokens, self.head_num, self.head_size, dtype=torch.float16)
        cache_k = torch.rand(self.block_num, self.block_size, self.head_num, self.head_size, dtype=torch.float16)
        cache_v = torch.rand(self.block_num, self.block_size, self.head_num, self.head_size_v, dtype=torch.float16)
        block_tables = torch.randint(0, self.block_num, 
                                    size=(ntokens, self.max_block_nums_per_query), dtype=torch.int32)
        context_lens = torch.randint(1, self.max_context_len, size=(ntokens, ), dtype=torch.int32)

        out_tensor = torch.zeros(ntokens, self.head_num, self.head_size_v, dtype=torch.float16)

        bind = {'in4': context_lens.cpu()}
        inputs = {
            'in0': q.npu(), 
            'in1': cache_k.npu(), 
            'in2': cache_v.npu(), 
            'in3': block_tables.npu(), 
            'in4': context_lens.npu()
        }
        outputs = {'out0': out_tensor.npu()}

        self.run_compare(self.op_set, inputs, outputs, bind)


if __name__ == '__main__':
    unittest.main()
