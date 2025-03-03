# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelEmbedding,
    load_column_multi,
)
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.quantize.pack_type import PackType


class QwenRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class QwenRMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class QwenRMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.ori = QwenRMSNorm(prefix, weights, eps)
        self.anti = QwenRMSNormBias(f'{prefix}.module', weights, eps)


class QwenRMSNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.ori = QwenRMSNorm(f'{prefix}.ori', weights, eps)
        self.anti = QwenRMSNormBias(f'{prefix}.anti', weights, eps)


class FlashQwenAttention(nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads  # for qwen1.5 and qwen2,the value is 128
        self.rank = weights.process_group.rank()
        
        if config.num_key_value_heads < weights.process_group.size():
            repeat_times = weights.process_group.size() // config.num_key_value_heads
        else:
            repeat_times = 1

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size() # 4
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads * repeat_times
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads
        
        self.pack_type = PackType.ALL_FP
        
        if config.num_experts_per_tok == 8:  # qwen2
            self.pre_norm = self.load_pre_norm(prefix, weights).npu()
            self.qkv_weight, self.qkv_bias = self.pack_query_key_value(prefix, weights)
            self.o_proj = self.load_o_proj(f"{prefix}.o_proj", weights)
        else:  # qwen1.5 model.layers.4.self_attn.q_proj.bias
            self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size,
                bias=True,
            )
            
            self.o_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                gqa_size=self.head_size,
                bias=False,
            )
        self.prefix = prefix
    
    def load_pre_norm(self, prefix, weights):
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_prefix = f"{layer_prefix}.input_layernorm.weight"
        norm_weight = weights.get_tensor(norm_prefix)
        return norm_weight
    
    def pack_query_key_value(self, prefix, weights):
        full_query = weights.get_tensor(f"{prefix}.q_proj.weight")
        padding_weights_zero = torch.zeros(
            size=(self.head_size, self.hidden_size),
            dtype=full_query.dtype,
            device=full_query.device
        )
        padding_bias_zero = torch.zeros(size=(self.head_size,), dtype=full_query.dtype, device=full_query.device)
        
        q_proj_weight, q_proj_bias = self.load_query(
            prefix=f"{prefix}.q_proj",
            weights=weights,
            padding_weights_zero=padding_weights_zero,
            padding_bias_zero=padding_bias_zero
        )
        
        k_proj_weight, k_proj_bias = self.load_key_value(prefix=f"{prefix}.k_proj", weights=weights)
        v_proj_weight, v_proj_bias = self.load_key_value(prefix=f"{prefix}.v_proj", weights=weights)
        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight)).npu()
        qkv_bias = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias)).npu()
        return qkv_weight, qkv_bias


    def load_query(self, prefix, weights, padding_weights_zero, padding_bias_zero):
        full_weight = weights.get_tensor(f"{prefix}.weight")
        full_bias = weights.get_tensor(f"{prefix}.bias")
        odd_rank_hidden_size = self.head_size * (self.num_heads - 1)
        even_rank_hidden_size = self.head_size * self.num_heads
        if self.rank % 2 == 0:
            start = (self.rank // 2) * (odd_rank_hidden_size + even_rank_hidden_size)
            part_weight = full_weight[start:(start + even_rank_hidden_size), :]
            part_bias = full_bias[start:(start + even_rank_hidden_size)]
        else:
            start = ((self.rank + 1) / 2) * even_rank_hidden_size + ((self.rank + 1) / 2 - 1) * odd_rank_hidden_size
            start = int(start)
            part_weight = full_weight[start:(start + odd_rank_hidden_size), :]
            part_weight = torch.cat((part_weight, padding_weights_zero), dim=0)
            part_bias = full_bias[start:(start + odd_rank_hidden_size)]
            part_bias = torch.cat((part_bias, padding_bias_zero), dim=0)
        return part_weight, part_bias
    
    
    def load_key_value(self, prefix, weights):
        per_rank_hidden_size = self.head_size
        start = self.rank // 2
        full_weight = weights.get_tensor(f"{prefix}.weight")
        full_bias = weights.get_tensor(f"{prefix}.bias")
        part_weights = \
        full_weight[start * per_rank_hidden_size : start * per_rank_hidden_size + per_rank_hidden_size, :]
        part_bias = full_bias[start * per_rank_hidden_size : start * per_rank_hidden_size + per_rank_hidden_size]
        return part_weights, part_bias
    
    def load_o_proj(self, prefix, weights):
        full_weight = weights.get_tensor(f"{prefix}.weight")
        padding_weights_zero = torch.zeros(
                                size=(self.head_size, self.hidden_size),
                                dtype=full_weight.dtype,
                                device=full_weight.device)
        odd_rank_hidden_size = self.head_size * (self.num_heads - 1)
        even_rank_hidden_size = self.head_size * self.num_heads
        if self.rank % 2 == 0:
            start = (self.rank // 2) * (odd_rank_hidden_size + even_rank_hidden_size)
            part_weight = full_weight[:, start:(start + even_rank_hidden_size)]
        else:
            start = ((self.rank + 1) / 2) * even_rank_hidden_size + ((self.rank + 1) / 2 - 1) * odd_rank_hidden_size
            start = int(start)
            part_weight = full_weight[:, start:(start + odd_rank_hidden_size)]
            part_weight = torch.cat((part_weight, padding_weights_zero.T), dim=1)
        return part_weight.npu()
        
        
class QwenMLP(nn.Module):
    def __init__(self, prefix, config, weights, intermediate_size):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = load_column_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            head_size=1,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",  # down_proj
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class QwenEp(nn.Module):
    """
    for experts parallel.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        expert_gate_proj = weights.get_tensor(f"{prefix}.gate_proj.weight")
        self.expert_gate_proj = nn.Parameter(expert_gate_proj)
        expert_up_proj = weights.get_tensor(f"{prefix}.up_proj.weight")
        self.expert_up_proj = nn.Parameter(expert_up_proj)
        expert_down_proj = weights.get_tensor(f"{prefix}.down_proj.weight")
        self.expert_down_proj = nn.Parameter(expert_down_proj)


class QwenExpertGate(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)


class QwenSharedExpertGate(nn.Module):

    def __init__(self, prefix, config, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)


class QwenMoe(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = True  # default the model is tensor parallel
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        if self.expert_parallel_degree == 0:
            msg = "expert parallel degree should not be 0!"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        expert_per_rank = config.num_experts / self.expert_parallel_degree

        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        # gate
        gate_prefix = f"{prefix}.gate"
        self.gate = QwenExpertGate(prefix=gate_prefix, config=config, weights=weights)
        # common experts
        expert_prefix = f"{prefix}.experts"
        if self.tp:
            self.experts = nn.ModuleList([QwenMLP(prefix=f"{expert_prefix}.{i}", config=config, weights=weights,
                                                  intermediate_size=config.moe_intermediate_size)
                                          for i in range(config.num_experts)])
        else:
            self.experts = nn.ModuleList()
            for j in range(config.num_experts):
                if j < expert_per_rank:
                    expert_id = int(j + self.tp_rank * expert_per_rank)
                    self.experts.append(QwenEp(prefix=f"{expert_prefix}.{expert_id}", config=config, weights=weights))
        # share experts
        shared_expert_prefix = f"{prefix}.shared_expert"
        self.shared_expert = QwenMLP(prefix=shared_expert_prefix, config=config, weights=weights,
                                     intermediate_size=config.shared_expert_intermediate_size)
        # share experts gate
        shared_expert_gate_prefix = f"{prefix}.shared_expert_gate"
        self.shared_expert_gate = QwenSharedExpertGate(prefix=shared_expert_gate_prefix, config=config, weights=weights)


class FlashQwenLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        
        self.self_attn = FlashQwenAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        
        self.mlp = QwenMoe(prefix=f"{prefix}.mlp", config=config, weights=weights)
        
        self.input_layernorm = QwenRMSNorm(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        
        self.post_attention_layernorm = QwenRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )


class FlashQwenModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashQwenLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = QwenRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

