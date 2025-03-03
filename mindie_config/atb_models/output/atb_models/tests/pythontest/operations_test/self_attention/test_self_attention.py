# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import random
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestSelfAttentionOperation(operation_test.OperationTest):
    def setUp(self):
        self.batch = random.randint(2, 8)
        self.layer = random.randint(2, 8)
        self.head_num = random.randint(2, 8)
        self.head_size = 128
        self.head_size_v = 16 * random.randint(1, 8)
        self.hidden_size = self.head_num * self.head_size
        self.hidden_size_v = self.head_num * self.head_size_v

        self.q_scale = 0.2
        self.qk_scale = 1

        self.min_seqlen, self.max_seqlen = 1, 5
        self.min_token_offset_start, self.max_token_offset_start = 0, 5

        self.op_type = "SelfAttention"
        self.op_name = "SelfAttentionOperation"
        self.op_param = {
            "headNum": self.head_num,
            "qScale": float(self.q_scale),
            "qkScale": float(self.qk_scale),
            "maskType": "MASK_TYPE_NORM"
        }
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensors):
        q, k, v, cache_k, cache_v, attention_mask, token_offset, seq_len, layerid = \
            in_tensors['in0'], in_tensors['in1'], in_tensors['in2'], in_tensors['in3'], \
            in_tensors['in4'], in_tensors['in5'], in_tensors['in6'], in_tensors['in7'], in_tensors['in8']

        layerid = int(layerid[0])
        data_type = q.dtype
        context_list = []
        seq_len = seq_len.tolist()
        token_offset = token_offset.tolist()
        seq_offset = 0
        for i, _ in enumerate(range(self.batch)):
            cur_seqlen = seq_len[i]
            cur_token_offset = token_offset[i]
            cur_token_offset_start = cur_token_offset - cur_seqlen
            cur_q = q[seq_offset:seq_offset + cur_seqlen]
            cur_k = k[seq_offset:seq_offset + cur_seqlen]
            cur_v = v[seq_offset:seq_offset + cur_seqlen]
            if cur_token_offset_start > 0:
                past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                cur_k = torch.concat([past_k, cur_k], dim=0)
                cur_v = torch.concat([past_v, cur_v], dim=0)
            cur_q = (cur_q * self.q_scale).view(cur_seqlen, self.head_num, self.head_size).transpose(0, 1)
            cur_k = cur_k.view(cur_token_offset, self.head_num, self.head_size).permute(1, 2, 0)
            cur_qk = torch.bmm(cur_q, cur_k)    # [head_num, seq_len, token_offset]

            if attention_mask.ndim == 3:  # masked_fill
                cur_qk = cur_qk + \
                    attention_mask[i, cur_token_offset_start:cur_token_offset_start + cur_seqlen, :cur_token_offset]
            else:
                cur_qk = cur_qk + \
                    attention_mask[cur_token_offset_start:cur_token_offset_start + cur_seqlen, :cur_token_offset]
            cur_qk = cur_qk * self.qk_scale
            cur_qk = torch.nn.functional.softmax(cur_qk.type(torch.float32), dim=-1).type(data_type)

            cur_v = cur_v.view(cur_token_offset, self.head_num,
                               self.head_size_v).transpose(0, 1)
            cur_context = torch.bmm(cur_qk, cur_v).transpose(
                0, 1).contiguous().view(cur_seqlen, self.head_num * self.head_size_v)
            context_list.append(cur_context)

            seq_offset = seq_offset + cur_seqlen

        context = torch.concat(context_list, dim=0)
        return [context]

    def test_float16(self):

        seq_len = torch.randint(self.min_seqlen, self.max_seqlen, (self.batch,), dtype=torch.int32)
        token_offset_start = torch.randint(
            self.min_token_offset_start, self.max_token_offset_start, (self.batch,), dtype=torch.int32)
        token_offset = token_offset_start + seq_len
        total_seqlen = self.max_token_offset_start + self.max_seqlen
        ntokens = int(seq_len.sum())

        q = torch.rand(ntokens, self.hidden_size, dtype=torch.float16)
        k = torch.rand(ntokens, self.hidden_size, dtype=torch.float16)
        v = torch.rand(ntokens, self.hidden_size_v, dtype=torch.float16)
        cache_k = torch.rand(self.layer, self.batch, total_seqlen, self.hidden_size, dtype=torch.float16)
        cache_v = torch.rand(self.layer, self.batch, total_seqlen, self.hidden_size_v, dtype=torch.float16)

        attention_mask = torch.zeros(self.batch, total_seqlen, total_seqlen, dtype=torch.float16)
        layerid = torch.randint(self.layer, (1,), dtype=torch.int32)

        out_tensor = torch.zeros(ntokens, self.hidden_size_v, dtype=torch.float16)

        bind_map = {'in6': token_offset.cpu(), 'in7': seq_len.cpu()}
        inputs = {
            'in0': q.npu(), 
            'in1': k.npu(), 
            'in2': v.npu(), 
            'in3': cache_k.npu(), 
            'in4': cache_v.npu(), 
            'in5': attention_mask.npu(), 
            'in6': token_offset.npu(), 
            'in7': seq_len.npu(), 
            'in8': layerid.npu()
        }
        outputs = {'out0': out_tensor.npu()}

        self.run_compare(self.op_set, inputs, outputs, bind_map)


if __name__ == '__main__':
    unittest.main()
