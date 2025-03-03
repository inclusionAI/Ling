# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
from typing import List

import torch
from atb_llm.utils.log import logger
from torch import nn

from .fast_linear import FastLinear
from ...quantize.quant_type import QuantType
from ...quantize.w4a16 import W4A16LinearStatic
from ...quantize.w8a16 import W8A16LinearStatic
from ...quantize.w8a8 import W8A8LinearStatic
from ...quantize.w8a8sc import W8A8SparseCompressedLinear
from ...quantize.w8a8_dynamic import W8A8DynamicLinearStatic


def get_linear(weight, bias, quantize, is_norm=False, **kwargs):
    if quantize is None:
        linear = FastLinear(weight, bias, is_norm)
    elif quantize in [QuantType.W8A8, QuantType.W8A8S]:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, deq_scale, quant_bias, input_scale, input_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a8` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8LinearStatic(
                weight=qweight,
                deq_scale=deq_scale,
                input_scale=input_scale,
                quant_bias=quant_bias,
                input_offset=input_offset,
                bias=bias
            )
    elif quantize == QuantType.W4A16:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w4a16` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W4A16LinearStatic(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                bias=bias
            )
    elif quantize == QuantType.W8A16:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a16` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A16LinearStatic(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                bias=bias
            )
    elif quantize == QuantType.W8A8SC:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, deq_scale, quant_bias, input_scale, input_offset, index = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a8sc` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8SparseCompressedLinear(
                weight=qweight,
                deq_scale=deq_scale,
                input_scale=input_scale,
                quant_bias=quant_bias,
                input_offset=input_offset,
                index=index
            )
    elif quantize == QuantType.W8A8_DYNAMIC:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a8 dynamic` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8DynamicLinearStatic(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                bias=bias
            )
    else:
        raise AssertionError(
            f"Quantization `{quantize}` is not implemented yet. "
            f"此类型从权重文件config.json中的`quantize`字段中获取。"
            f"若非量化权重，config.json中无需配置此字段；"
            f"若为量化权重，当前支持的量化类型为`w4a16`，`w8a16`，`w8a8`，`w8a8s`和`w8a8sc`。"
        )
    
    # 更新Linear metainfo
    linear.prefixes = kwargs.get("prefixes", [])
    linear.num_linear_before_pack = kwargs.get("num_linear_before_pack", 1)
    linear.tensor_parallel_dim = kwargs.get("tensor_parallel_dim", 0)
    linear.align_size = kwargs.get("align_size", 1)
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, input_tensor):
        return self.linear.forward(input_tensor)


class TensorHead(SuperLayer):
    def __init__(self, linear):
        super().__init__(linear)

    @staticmethod
    def load_weight(config, prefix: str, weights, is_norm=False):
        weight = weights.get_whole_tensor(f"{prefix}.weight", dim=0)

        # GPTQ doesn't quantize heads (nor embeddings)
        if config.quantize == "gptq":
            quantize = None
        else:
            quantize = config.quantize
        return TensorHead(
            get_linear(weight, bias=None, quantize=quantize, is_norm=is_norm),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        out = torch.mm(input_tensor, self.linear.weight.T)
        return out


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group, should_gather: bool):
        super().__init__(linear)
        self.process_group = process_group
        self.should_gather = should_gather

    @staticmethod
    def load_weight(config, prefix: str, weights, is_norm=False):
        weight = weights.get_tensor(f"{prefix}.weight")
        should_gather = False
        # GPTQ doesn't quantize heads (nor embeddings)
        quantize = None if config.quantize == "gptq" else config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize, is_norm=is_norm),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    @staticmethod
    def load(config, prefix: str, weights, is_norm=False):
        should_gather = True
        if weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        # GPTQ doesn't quantize heads (nor embeddings)
        quantize = None if config.quantize == "gptq" else config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize, is_norm=is_norm),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not self.should_gather:
            return super().forward(input_tensor)

        world_size = self.process_group.size()
        if len(input_tensor.shape) == 2 and isinstance(self.linear, FastLinear):
            out_dim = self.linear.weight.shape[0]
            if input_tensor.shape[0] == 1:
                world_out = input_tensor.new_empty(1, out_dim * world_size)
                local_out = input_tensor.new_empty(1, out_dim)
                gather_input = local_out
            else:
                world_out = input_tensor.new_empty(out_dim * world_size, input_tensor.shape[0])
                gather_input = input_tensor.new_empty(out_dim, input_tensor.shape[0])
                local_out = gather_input.T

            torch.mm(input_tensor, self.linear.weight.T, out=local_out)
            torch.distributed.all_gather_into_tensor(
                world_out, gather_input, group=self.process_group
            )

            if input_tensor.shape[0] == 1:
                return world_out
            return world_out.T

        output = super().forward(input_tensor)
        world_output = [
            torch.empty_like(output)
            for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_qkv(cls, config, prefix: str, weights, bias: bool, hidden_size, num_heads, num_kv_heads=None, dim=0, padding=False):
        """Specific method when the QKV was joined after the fact"""
        if num_kv_heads is None:
            num_kv_heads = num_heads
        weight = weights.get_weights_col_packed_qkv(
            prefix, quantize=config.quantize, hidden_size=hidden_size, 
            num_heads=num_heads, num_kv_heads=num_kv_heads, dim=dim,padding=padding
        )
        if bias:
            bias = weights.get_tensor_col_packed_qkv(
                f"{prefix}.bias", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, prefixes=[prefix], num_linear_before_pack=3,
            tensor_parallel_dim=dim, align_size=hidden_size // num_heads,
        )
        return cls(linear)

    @classmethod
    def load_gate_up(cls, config, prefix: str, weights, bias: bool):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_mlp(
            prefix, quantize=config.quantize
        )
        if bias:
            bias = weights.get_tensor_col_packed_mlp(f"{prefix}.bias")
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, prefixes=[prefix], num_linear_before_pack=2,
            tensor_parallel_dim=1, align_size=1
        )
        return cls(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool, dim=0):
        return cls.load_multi(config, [prefix], weights, bias, dim=dim)

    @classmethod
    def load_multi(cls, config, prefixes: List[str], weights, bias: bool, dim: int):
        weight = weights.get_multi_weights_col(
            prefixes, quantize=config.quantize, dim=dim
        )

        if bias:
            if config.quantize == QuantType.W8A8SC:
                b = [weights.get_tensor(f"{p}.bias") for p in prefixes]
            else:
                b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=0)
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, prefixes=prefixes, num_linear_before_pack=len(prefixes),
            tensor_parallel_dim=dim, align_size=1
        )
        return cls(linear)

    @classmethod
    def load_o(cls, config, prefix: str, weights, bias: bool, hidden_size, num_heads, num_kv_heads=None):
        """Specific method when the QKV was joined after the fact"""
        if num_kv_heads is None:
            num_kv_heads = num_heads
        weight = weights.get_weights_col_packed_o(
            prefix, quantize=config.quantize, hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        if bias:
            bias = weights.get_tensor_col_packed_o(
                f"{prefix}.bias", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, prefixes=[prefix], num_linear_before_pack=3,
            tensor_parallel_dim=0, align_size=hidden_size // num_heads,
        )
        return cls(linear)

    @classmethod
    def load_column_multi_c(cls, config, prefixes: List[str], weights, hidden_size, num_heads, num_kv_heads=None):
        """Specific method when the QKV was joined after the fact"""
        if num_kv_heads is None:
            num_kv_heads = num_heads
        weight_q = weights.get_weights_col_packed_q(
            prefixes[0], quantize=config.quantize, hidden_size=hidden_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads)
        bias_q = weights.get_tensor_col_packed_q(
            f"{prefixes[0]}.bias", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
        weight_k = weights.get_weights_col_packed_k(
            prefixes[1], quantize=config.quantize, hidden_size=hidden_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads)
        bias_k = weights.get_tensor_col_packed_k(
            f"{prefixes[1]}.bias", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
        weight_v = weights.get_weights_col_packed_v(
            prefixes[2], quantize=config.quantize, hidden_size=hidden_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads)
        bias_v = weights.get_tensor_col_packed_v(
            f"{prefixes[2]}.bias", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
        weight = torch.cat([weight_q, weight_k, weight_v], dim=0)
        bias = torch.cat([bias_q, bias_k, bias_v], dim=0)
        linear = get_linear(
            weight, bias, config.quantize, prefixes=prefixes, num_linear_before_pack=len(prefixes),
            tensor_parallel_dim=0, align_size=hidden_size // num_heads
        )
        return cls(linear)


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool, bias_pre_add=False, gqa_size=1, dim=1):
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize, gqa_size=gqa_size, dim=dim)
        if bias and bias_pre_add:
            bias = weights.get_tensor(f"{prefix}.bias")
        elif bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(
                weight, bias, config.quantize, prefixes=[prefix],
                tensor_parallel_dim=dim, align_size=gqa_size
            ),
            process_group=weights.process_group,
        )
    
    @classmethod
    def load_att_dense(cls, config, prefix: str, weights, bias: bool, bias_pre_add=False, gqa_size=1, dim=1):
        weight = weights.get_multi_weights_row_att_dense(prefix, quantize=config.quantize, gqa_size=gqa_size, dim=dim)
        if bias and bias_pre_add:
            bias = weights.get_tensor(f"{prefix}.bias")
        elif bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(
                weight, bias, config.quantize, prefixes=[prefix],
                tensor_parallel_dim=dim, align_size=gqa_size
            ),
            process_group=weights.process_group,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = super().forward(input_tensor)
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorReplicatedLinear(SuperLayer):
    def __init__(self, linear):
        super().__init__(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_replicated_weights(prefix, quantize=config.quantize)
        if bias :
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None

        return cls(get_linear(weight, bias, config.quantize, prefixes=[prefix]))
