from typing import List

import torch
import torch_npu
import torch.distributed
from torch.nn import functional as F

from atb_llm.utils.log import logger
from .attention import AttentionMask, flash_attn, paged_attn, reshape_and_cache, KvCache, FA3
from .embedding.position_rotary_embedding import PositionRotaryEmbedding
from .embedding.tensor_embedding import TensorEmbedding, TensorParallelEmbedding
from .linear import (
    get_linear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorReplicatedLinear,
    TensorParallelHead,
    TensorHead
)
from .linear.reduce_quant import ReduceQuant
from .norm.fast_layer_norm import RMSNorm, RMSNormBias, RMSNormWrapper, RMSNormAntiOutlierWrapper


def _load_gqa(config, prefix: str, weights):
    hidden_size, num_attention_heads = config.hidden_size, config.num_attention_heads
    process_group_size = weights.process_group.size()
    if not hidden_size % num_attention_heads == 0:
        logger.error(f'{hidden_size} % {num_attention_heads} != 0')
    if not num_attention_heads % process_group_size == 0:
        logger.error(f'{num_attention_heads} % {process_group_size} != 0')

    weight_prefixes = [f"{prefix}.{proj}" for proj in ["q_proj", "k_proj", "v_proj"]]
    weight = weights.get_multi_weights_col(prefixes=weight_prefixes, quantize=config.quantize, dim=0)

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize, prefixes=weight_prefixes,
        num_linear_before_pack=len(weight_prefixes), tensor_parallel_dim=0, align_size=1)
    )


def load_column_multi(
        config, prefixes: List[str], weights, head_size, lm_head: bool = False, \
        norm: bool = False, bias: bool = False, dim: int = 0, norm_head: bool = False,
):
    soc_version = torch_npu._C._npu_get_soc_version()
    quantize = None if lm_head else config.quantize
    weight = weights.get_multi_weights_col(prefixes, quantize=quantize, dim=0, gqa_size=head_size, norm_head=norm_head)
    if bias:
        b = [weights.get_sharded(f"{p}.bias", dim=0, gqa_size=head_size) for p in prefixes]
        bias = torch.cat(b, dim=dim)
    else:
        bias = None
    if lm_head:
        weight_type = weight.dtype
        weight = weight.float()
        weight = weight if not norm else torch.nan_to_num(F.normalize(weight))
        if soc_version == 240:
            weight = weight.to(dtype=weight_type)
            weight = weight.npu()
        else:
            weight = weight.to(dtype=weight_type).npu()
    linear = get_linear(weight, bias, quantize, prefixes=prefixes,
        num_linear_before_pack=len(prefixes), tensor_parallel_dim=0, align_size=head_size)

    process_group = weights.process_group
    should_gather = weights.process_group.size() != 1
    if lm_head:
        return TensorParallelHead(linear, process_group=process_group, should_gather=should_gather)
    else:
        return TensorParallelColumnLinear(linear)


def load_row(config, prefix: str, weights, head_size):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1, gqa_size=head_size)
    linear = get_linear(weight, None, quantize=config.quantize, prefixes=[prefix],
        tensor_parallel_dim=1, align_size=head_size)
    return TensorParallelRowLinear(linear, process_group=weights.process_group)