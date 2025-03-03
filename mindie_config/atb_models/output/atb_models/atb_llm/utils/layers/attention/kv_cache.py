# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from collections import OrderedDict

import torch
from torch import nn
import numpy as np
from atb_llm.utils import OpBackend
from atb_llm.utils.initial import NPUSocInfo


class KvCache(nn.Module):
    soc_info = None

    def __init__(self, scale: list, offset: list, backend: OpBackend, weights):
        super().__init__()
        self.prefix = None

        k_scale, v_scale = scale
        k_offset, v_offset = offset
        # FP16和BF16都先cast成FP32，按位转成INT32后
        k_offset = torch.from_numpy(
            np.frombuffer(k_offset.to(torch.float32).numpy().tobytes(), dtype=np.int32).copy())
        v_offset = torch.from_numpy(
            np.frombuffer(v_offset.to(torch.float32).numpy().tobytes(), dtype=np.int32).copy())
        self.k_quant_offset = nn.Parameter(k_offset.to(torch.int8), requires_grad=False)
        self.v_quant_offset = nn.Parameter(v_offset.to(torch.int8), requires_grad=False)
        self.k_quant_scale = nn.Parameter(k_scale, requires_grad=False)
        self.v_quant_scale = nn.Parameter(v_scale, requires_grad=False)
        if backend == OpBackend.ATB:
            # FP16先cast成FP32，按位转成INT32后，再转成INT64，算子需要INT64；BF16转成FLOAT32，算子需要FLOAT32；其余情况不支持
            if weights.dtype == torch.float16:
                k_dequant_scale = torch.from_numpy(
                    np.frombuffer(k_scale.to(torch.float32).numpy().tobytes(), dtype=np.int32).astype(np.int64))
                v_dequant_scale = torch.from_numpy(
                    np.frombuffer(v_scale.to(torch.float32).numpy().tobytes(), dtype=np.int32).astype(np.int64))
            elif weights.dtype == torch.bfloat16:
                k_dequant_scale = k_scale.to(torch.float32)
                v_dequant_scale = v_scale.to(torch.float32)
            else:
                raise ValueError("The weight type must be FP16 or BF16.")
            self.k_dequant_scale = nn.Parameter(k_dequant_scale, requires_grad=False)
            self.v_dequant_scale = nn.Parameter(v_dequant_scale, requires_grad=False)
            self.k_dequant_offset = nn.Parameter(k_offset.to(torch.int32), requires_grad=False)
            self.v_dequant_offset = nn.Parameter(v_offset.to(torch.int32), requires_grad=False)
        elif backend == OpBackend.ACLNN:
            self.k_dequant_scale = nn.Parameter(torch.cat(scale, dim=-1).view(2, -1), requires_grad=False)
            self.v_dequant_scale = self.k_dequant_scale
            self.k_dequant_offset = nn.Parameter(torch.cat(offset, dim=-1).view(2, -1), requires_grad=False)
            self.v_dequant_offset = self.k_dequant_offset

    @classmethod
    def load(cls, prefix_k, prefix_v, weights, gqa_size: int = 1, backend=OpBackend.ATB):
        # 原始权重是FP16 or BF16
        k_scale = weights.get_sharded(f"{prefix_k}.kv_cache_scale", dim=0, gqa_size=gqa_size)
        k_offset = weights.get_sharded(f"{prefix_k}.kv_cache_offset", dim=0, gqa_size=gqa_size)
        v_scale = weights.get_sharded(f"{prefix_v}.kv_cache_scale", dim=0, gqa_size=gqa_size)
        v_offset = weights.get_sharded(f"{prefix_v}.kv_cache_offset", dim=0, gqa_size=gqa_size)
        return cls([k_scale, v_scale], [k_offset, v_offset], backend, weights)

    @classmethod
    def set_soc_info(cls):
        cls.soc_info = NPUSocInfo()

    @classmethod
    def weight_format_cast(cls, tensor):
        if not cls.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    def get_weights(self, prefix):
        self.prefix = prefix
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.k_quant_scale"] = self.weight_format_cast(self.k_quant_scale.data)
        weight_dict[f"{prefix}.k_dequant_scale"] = self.weight_format_cast(self.k_dequant_scale.data)
        weight_dict[f"{prefix}.k_quant_offset"] = self.weight_format_cast(self.k_quant_offset.data)
        weight_dict[f"{prefix}.k_dequant_offset"] = self.weight_format_cast(self.k_dequant_offset.data)
        weight_dict[f"{prefix}.v_quant_scale"] = self.weight_format_cast(self.v_quant_scale.data)
        weight_dict[f"{prefix}.v_dequant_scale"] = self.weight_format_cast(self.v_dequant_scale.data)
        weight_dict[f"{prefix}.v_quant_offset"] = self.weight_format_cast(self.v_quant_offset.data)
        weight_dict[f"{prefix}.v_dequant_offset"] = self.weight_format_cast(self.v_dequant_offset.data)
        return weight_dict
