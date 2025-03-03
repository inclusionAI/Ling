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
import math
import json
from abc import abstractmethod
from collections import OrderedDict

import torch
from torch import nn
from transformers.modeling_utils import PretrainedConfig

import _libatb_torch as atb
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, OperationBackend, NormType, ActivationType
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import AttnType
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.models.base.model_utils import AttnLinearInfo, MlpLinearInfo, LmHeadLinearInfo, LinearInfo
from atb_llm.utils.quantize.quant_type import is_same_type, QuantType, LinearTypeV2
from atb_llm.utils.weights import Weights
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    TensorParallelEmbedding,
    load_column_multi,
)


class AntiModule(nn.Module):
    """
    Anti-outlier module.

    Args:
        prefix (str): The prefix of the module.
        weights (Weights): The weights of the module.
    """
    def __init__(self, prefix: str, weights: Weights):
        super().__init__()

        # 模型结构
        weight = weights.get_tensor(f"{prefix}.module.weight")
        self.weight = nn.Parameter(weight)
        bias = weights.get_tensor(f"{prefix}.module.bias")
        self.bias = nn.Parameter(bias)

    def get_weights(self, prefix, idx=0) -> OrderedDict:
        """Get weight according to prefix, save it to a dict."""
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight" if idx == 0 else f"{prefix}_{idx}.weight"] = self.weight.data
        weights_dict[f"{prefix}.bias" if idx == 0 else f"{prefix}_{idx}.bias"] = self.bias.data
        return weights_dict


class BaseRMSNorm(nn.Module):
    """
    Base RMSNorm layer module.

    Args:
        config (PretrainedConfig): PretrainedConfig instance.
        prefix (str): Prefix of the layer.
        weights (Weights): Weights instance.
        linear_info (LinearInfo): LinearInfo instance of the layer.
    """
    def __init__(self, prefix: str, config: PretrainedConfig, weights: Weights, linear_info: LinearInfo):
        super().__init__()

        # 配置信息
        self.prefix = prefix
        self.config = config
        self.quantize = config.quantize
        self.linear_info = linear_info
        self.is_antioutlier = False
        self.has_bias = False
        self.is_pack = linear_info.is_pack
        self.split_num = linear_info.split_num
        self.linear_modules = [None]
        if self.is_pack:
            self.linear_modules = [self.linear_info.pack_linear]
        else:
            if isinstance(self.linear_info, AttnLinearInfo):
                self.linear_modules = [self.linear_info.q_linear, self.linear_info.k_linear, self.linear_info.v_linear]
            if isinstance(self.linear_info, MlpLinearInfo):
                self.linear_modules = [self.linear_info.gate_linear, self.linear_info.up_linear]
        
        if isinstance(self.linear_info, LmHeadLinearInfo):
            self.quantize = QuantType.FLOAT
        # 模型结构
        self.module = None
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        
        try:
            self.module = AntiModule(prefix, weights)
            self.is_antioutlier = True
        except AssertionError:
            self.module = None
        
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
            self.bias = nn.Parameter(bias)
            self.has_bias = True
        except AssertionError:
            if self.quantize == QuantType.W8A8SC or self.quantize == QuantType.W8A8:
                bias = torch.zeros(weight.shape, dtype=weights.dtype)
                self.bias = nn.Parameter(bias)
                self.has_bias = True
            else:
                self.bias = None

    def get_weights(self, prefix: str) -> OrderedDict:
        """Get the weights of the layer."""
        weights_dict = OrderedDict()
        for i in range(self.split_num):
            weights_dict[f"{prefix}.weight" if i == 0 else f"{prefix}_{i}.weight"] = self.weight.data
            if self.has_bias:
                weights_dict[f"{prefix}.bias" if i == 0 else f"{prefix}_{i}.bias"] = self.bias.data

        if self.is_antioutlier:
            for i in range(self.split_num):
                weights_dict.update(self.module.get_weights(f"{prefix}.module", i))
        return weights_dict

    def build_graph(self, graph: AtbGraph, is_prefill: bool):
        """Build compute graph for BaseRmsNorm layer."""
        for i in range(self.split_num):
            quant_type = "QUANT_UNDEFINED"
            if self.linear_modules[0] is not None:
                if self.linear_modules[i].linear_desc in [LinearTypeV2.W8A8, LinearTypeV2.W8A8S, LinearTypeV2.W8A8SC]:
                    quant_type = "QUANT_INT8"
                    self.has_bias = True
                elif self.has_bias:
                    self.has_bias = False
            
            norm_op_param = {
                "op_name": "norm",
                "category": CommonOpBuilderType.NORM,
                "has_bias": self.has_bias,
                "enable_add_norm": False,
                "norm_type": NormType.RMSNORM,
                "linear_module": self.linear_modules[i],
                "norm_param": {
                    'layerType': 'RMS_NORM_NORM',
                    'normParam': {
                        'quantType': quant_type,
                        'epsilon': self.config.rms_norm_eps
                    }
                }
            }
            norm_tensor_map = {
                "input": 'hidden_states',
                "weight": f"{self.prefix}.weight" if i == 0 else f"{self.prefix}_{i}.weight",
                "norm_out": f"{self.prefix}_out" if i == 0 else f"{self.prefix}_out_{i}",
            }
            if self.has_bias:
                norm_tensor_map.update({"bias": f"{self.prefix}.bias" if i == 0 else f"{self.prefix}_{i}.bias"})
            if self.is_antioutlier:
                norm_tensor_map.update({
                    "weight": f"{self.prefix}.module.weight" if i == 0 else f"{self.prefix}.module_{i}.weight",
                    "bias": f"{self.prefix}.module.bias" if i == 0 else f"{self.prefix}.module_{i}.bias"})

            builder = CommonOpBuilderManager.get_builder(norm_op_param)
            graph = builder.build(graph, norm_tensor_map)


class BaseAttention(torch.nn.Module):
    """
    Base attention block module for all models.
    
    Args:
        config (PretrainedConfig): The configuration object for the model.
        weights (Weights): The weights object containing the model weights.
        prefix (str): The prefix for the attention block.
        norm_prefix (str): The prefix for the normalization layer.
        is_fa (bool, optional): Whether to use Flash Attention, defaults to Fasle.
        backend (ComputationBackend, optional): The communication backend to use in multi-card settings,
            defaults to CommunicationBackend.LCCL.
        bias (bool, optional): Whether to use bias in the attention layers, defaults to False.
    """
    def __init__(
            self,
            config: PretrainedConfig,
            weights: Weights,
            prefix: str,
            norm_prefix: str,
            is_fa: bool = False,
            backend: CommunicationBackend = CommunicationBackend.LCCL,
            bias: bool = False
    ):
        super().__init__()

        # 配置信息
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.quantize = config.quantize
        self.prefix = prefix
        self.is_fa = is_fa
        self.backend = backend
        self.bias = bias

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads_per_rank = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
                                            // self.tp_world_size

        self.linear_info = AttnLinearInfo()
        self.norm_prefix = norm_prefix

        # 模型结构
        # Query Key Value
        if config.quantize == QuantType.W8A8SC:
            self.query_key_value = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.query_key_value",
                weights=weights,
                bias=False,
            )
            self.linear_info.is_pack = True
            self.linear_info.pack_linear = self.query_key_value.linear
        else:
            query_linear_desc = weights.get_linear_quant_type(f'{prefix}.q_proj.weight')
            key_linear_desc = weights.get_linear_quant_type(f'{prefix}.k_proj.weight')
            value_linear_desc = weights.get_linear_quant_type(f'{prefix}.v_proj.weight')

            if is_same_type([query_linear_desc, key_linear_desc, value_linear_desc]):
                self.query_key_value = load_column_multi(
                    config,
                    prefixes=[f"{prefix}.q_proj",
                              f"{prefix}.k_proj",
                              f"{prefix}.v_proj"],
                    weights=weights,
                    head_size=self.head_size,
                    bias=self.bias
                )
                self.linear_info.is_pack = True
                self.linear_info.pack_linear = self.query_key_value.linear
                if query_linear_desc in [LinearTypeV2.FLOAT16, LinearTypeV2.BFLOAT16]:
                    self.linear_info.is_all_float = True
            else:
                self.q_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.q_proj",
                    weights=weights,
                    bias=False,
                )
                self.k_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.k_proj",
                    weights=weights,
                    bias=False,
                )
                self.v_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.v_proj",
                    weights=weights,
                    bias=False,
                )
                self.linear_info.is_pack = False
                self.linear_info.split_num = 3
                self.linear_info.q_linear = self.q_proj.linear
                self.linear_info.k_linear = self.k_proj.linear
                self.linear_info.v_linear = self.v_proj.linear

        # Dense
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        self.linear_info.dense_linear = self.o_proj.linear

    def get_weights(self, prefix:str) -> OrderedDict:
        """Get the weights for BaseAttention block based on the prefix."""
        weights_dict = OrderedDict()
        if self.linear_info.is_pack:
            weights_dict.update(self.query_key_value.linear.get_weights(f"{prefix}.query_key_value"))
        else:
            weights_dict.update(self.q_proj.linear.get_weights(f"{prefix}.q_proj"))
            weights_dict.update(self.k_proj.linear.get_weights(f"{prefix}.k_proj"))
            weights_dict.update(self.v_proj.linear.get_weights(f"{prefix}.v_proj"))
        weights_dict.update(self.o_proj.linear.get_weights(f"{prefix}.o_proj"))
        return weights_dict

    def reshape_q(self, org_shape: list | tuple) -> list:
        """Reshape query from [batch_size, **] to [batch_size, num_heads_pre_rank, head_size]."""
        return [org_shape[0], self.num_heads_pre_rank, self.head_size]

    def reshape_kv(self, org_shape: list | tuple) -> list:
        """Reshape key/value from [batch_size, **] to [batch_size, num_heads_per_rank, head_size]."""
        return [org_shape[0], self.num_key_value_heads_per_rank, self.head_size]

    def build_qkv_graph(self, graph: atb._GraphOperation):
        """Build qkv compute graph. In this method, get module according to the linear_info and build graph."""
        linear_modules = []
        if self.linear_info.is_pack:
            linear_modules = [self.linear_info.pack_linear]
        else:
            linear_modules = [self.linear_info.q_linear, self.linear_info.k_linear, self.linear_info.v_linear]
        
        qkv_linear_param = {
            "op_name": "qkv_split",
            "category": CommonOpBuilderType.QKV,
            "is_pack": self.linear_info.is_pack,
            "is_fa": self.is_fa,
            "head_dim": self.head_size,
            "head_num": self.num_heads_pre_rank,
            "kv_head_num":self.num_key_value_heads_per_rank,
            "linear_modules": linear_modules,
            "linear_param": {
                "op_name": "q_linear",
                "category": CommonOpBuilderType.LINEAR,
                "enable_quant_input": False,
                "default_dtype": self.dtype,
                "group_size": 128 if self.quantize == QuantType.W4A16 else 0
            }
        }
        qkv_linear_tensor_map = {
            "input": f'{self.norm_prefix}_out',
            "input_k": f'{self.norm_prefix}_out_1',
            "input_v": f'{self.norm_prefix}_out_2',
            "q_out": 'intermediate_q',
            "k_out": 'intermediate_k',
            "v_out": 'intermediate_v'
        }

        qkv_linear_builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        graph = qkv_linear_builder.build(graph, qkv_linear_tensor_map)

    def build_rope_graph(self, graph: atb._GraphOperation):
        """
        Build rope compute graph. In this method, initialize rope parameters and tensor map, 
            then build the rope op and graph.
        """
        rope_param = {
            "op_name": "rope",
            'head_num': self.num_heads_pre_rank,
            'kv_head_num': self.num_key_value_heads_per_rank,
            "category": CommonOpBuilderType.ROPE,
            "is_fa": self.is_fa,
            "atb_rope_param": {
                'rotaryCoeff': 2
            }
        }
        rope_tensor_map = {
            "q": 'intermediate_q',
            "k": 'intermediate_k',
            "cos_embedding": 'cos_embedding',
            "sin_embedding": 'sin_embedding',
            "seq_len": "seq_len",
            "q_out": 'intermediate_q',
            "k_out": 'intermediate_k',
        }
        rope_builder = CommonOpBuilderManager.get_builder(rope_param)
        graph = rope_builder.build(graph, rope_tensor_map)
    
    def build_dense_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """
        Build dense compute graph. In this method, initialize the dense linear layer parameters,
            then build the graph.
        """
        dense_linear_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.dense_linear,
            "enable_quant_input": True,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0
        }
        dense_linear_parallel_param = {
            "op_name": "dense_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": dense_linear_param,
            "enable_lcoc": True if is_prefill else False,
        }
        dense_linear_tensor_map = {
            "input": "attn_out",
            "linear_out": 'dense_out'
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(dense_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, dense_linear_tensor_map)

    def build_attention_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """
        Build attention compute graph: initialze the attention parameters and build operation and graph.
        """
        attention_param = {
            "op_name": "attention",
            "category": CommonOpBuilderType.ATTENTION,
            "is_prefill": is_prefill,
            "attn_type": AttnType.FLASH_ATTENTION if self.is_fa else AttnType.PAGED_ATTENTION,
            "head_size": self.head_size,
            "atb_reshape_and_cache_param": {},
            "operation_backend": OperationBackend.ATB,
            "atb_attention_param": self._get_atb_attention_param(is_prefill)
        }
        attention_tensor_map = self._get_attention_tensor_map()

        pa_attention_builder = CommonOpBuilderManager.get_builder(attention_param)
        graph = pa_attention_builder.build(graph, attention_tensor_map)

    @abstractmethod
    def build_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """Abstract method to build computation graph."""
        atten_res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                           op_name='atten_res_add')
        setattr(graph, 'atten_res_add', atten_res_add)

        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph, is_prefill)
        self.build_dense_graph(graph, is_prefill)
    
        graph.add_operation(graph.atten_res_add, ['hidden_states', 'dense_out'], ['hidden_states'])
    
    def _get_attention_tensor_map(self) -> dict:
        """Get tensor map for attention graph."""
        attention_tensor_map = {
            "q": "intermediate_q",
            "k": "intermediate_k",
            "v": "intermediate_v",
            "k_cache": "k_cache",
            "v_cache": "v_cache",
            "attention_mask": "attention_mask",
            "seq_len": "seq_len",
            "attention_out": "attn_out",
        }
        if self.is_fa:
            attention_tensor_map.update({
                "token_offset": "token_offset",
                "layer_id": "layer_id",
            })
        # PA
        else:
            attention_tensor_map.update({
                "slots": "slots_mapping",
                "block_tables": "block_tables",
            })
        return attention_tensor_map

    def _get_atb_attention_param(self, is_prefill: bool) -> dict:
        """Get attention parameters."""
        atb_attention_param = {
            'headNum': self.num_heads_pre_rank,
            'kvHeadNum': self.num_key_value_heads_per_rank,
            'qkScale': 1.0 / math.sqrt(self.head_size), 
        }
        if self.is_fa:
            atb_attention_param.update({
                'maskType': 'MASK_TYPE_NORM',
                'calcType': 'ENCODER' if is_prefill else 'DECODER'
            })
        # PA
        elif is_prefill:
            atb_attention_param.update({
                'maskType': 'MASK_TYPE_NORM',
                'calcType': 'PA_ENCODER',
                'isTriuMask': 1
            })
        return atb_attention_param


class BaseMLP(nn.Module):
    def __init__(self, prefix: str, config: PretrainedConfig, weights: Weights, 
                 norm_prefix: str, backend: CommunicationBackend = CommunicationBackend.LCCL):
        """
        Base implementation of MLP block.

        Args:
            prefix (str): Prefix string of the MLP block.
            config (PretrainedConfig): Configuration object.
            weights (Weights): Weights object.
            norm_prefix (str): Prefix string of the normalization layer.
            backend (CommunicationBackend, optional): Communication backend for multi-card computation.
                Defaults to `CommunicationBackend.LCCL`.
        """
        super().__init__()

        # 配置信息
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.quantize = config.quantize
        self.linear_info = MlpLinearInfo()
        self.norm_prefix = norm_prefix
        self.backend = backend

        # 模型结构
        # Gate Up
        if config.quantize == QuantType.W8A8SC:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False,
            )
            self.linear_info.is_pack = True
            self.linear_info.pack_linear = self.gate_up_proj.linear
        else:
            gate_linear_desc = weights.get_linear_quant_type(f'{prefix}.gate_proj.weight')
            up_linear_desc = weights.get_linear_quant_type(f'{prefix}.up_proj.weight')

            if is_same_type([gate_linear_desc, up_linear_desc]):
                self.gate_up_proj = load_column_multi(
                    config,
                    prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                    weights=weights,
                    head_size=1,
                )
                self.linear_info.is_pack = True
                self.linear_info.pack_linear = self.gate_up_proj.linear
                if gate_linear_desc in [LinearTypeV2.FLOAT16, LinearTypeV2.BFLOAT16]:
                    self.linear_info.is_all_float = True
            else:
                self.gate_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.gate_proj",
                    weights=weights,
                    bias=False,
                )
                self.up_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.up_proj",
                    weights=weights,
                    bias=False,
                )
                self.linear_info.is_pack = False
                self.linear_info.split_num = 2
                self.linear_info.gate_linear = self.gate_proj.linear
                self.linear_info.up_linear = self.up_proj.linear
        # Down
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.linear_info.down_linear = self.down_proj.linear

    def get_weights(self, prefix: str) -> OrderedDict:
        """Get gate/up/down linear weights."""
        weights_dict = OrderedDict()
        if self.linear_info.is_pack:
            weights_dict.update(self.gate_up_proj.linear.get_weights(f"{prefix}.gate_up_proj"))
        else:
            weights_dict.update(self.gate_proj.linear.get_weights(f"{prefix}.gate_proj"))
            weights_dict.update(self.up_proj.linear.get_weights(f"{prefix}.up_proj"))
        weights_dict.update(self.down_proj.linear.get_weights(f"{prefix}.down_proj"))
        return weights_dict

    def build_gateup_graph(self, graph: atb._GraphOperation):
        """Build gateup linear computation graph."""
        linear_param = {
            "op_name": "gate_up_linear",
            "category": CommonOpBuilderType.LINEAR,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0
        }
        if self.linear_info.up_weight_only:
            linear_param.update({"linear_module": self.linear_info.up_linear})
        elif self.linear_info.is_pack:
            linear_param.update({"linear_module": self.linear_info.pack_linear})
        else:
            linear_param.update({"linear_module": self.linear_info.gate_linear})

        gate_up_linear_param = {
            "op_name": "gate_up_linear",
            "category": CommonOpBuilderType.GATE_UP,
            "is_pack": self.linear_info.is_pack,
            "linear_param": linear_param
        }
        gate_up_linear_tensor_map = {
            "input": f'{self.norm_prefix}_out',
            "gate_up_out": 'gate_up_out'
        }
        if not self.linear_info.is_pack:
            gate_up_linear_param.update({"up_linear_param": {
                "op_name": "up_linear",
                "category": CommonOpBuilderType.LINEAR,
                "linear_module": self.linear_info.up_linear,
                "enable_quant_input": False,
                "default_dtype": self.dtype,
                "group_size": 128 if self.quantize == QuantType.W4A16 else 0
            }})
            gate_up_linear_tensor_map.update({"up_out": 'up_out'})

        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        graph = builder.build(graph, gate_up_linear_tensor_map)

    def build_activation_graph(self, graph: atb._GraphOperation):
        """Build activatoin function computeation graph."""
        act_param = {
            "op_name": "activation",
            "category": CommonOpBuilderType.ACTIVATION,
            "is_pack": self.linear_info.is_pack,
            "up_weight_only": self.linear_info.up_weight_only,
            "activation_type": ActivationType.SWIGLU if self.backend == CommunicationBackend.LCCL \
                else ActivationType.SWISH
        }
        act_tensor_map = {
            "input": 'gate_up_out',
            "act_out": 'mul_out'
        }
        if not self.linear_info.is_pack:
            act_tensor_map.update({"other_input": 'up_out'})
        act_builder = CommonOpBuilderManager.get_builder(act_param)
        graph = act_builder.build(graph, act_tensor_map)

    def build_down_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """Build down linear computation graph."""
        down_linear_param = {
            "op_name": "down_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.down_linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0
        }
        down_linear_tensor_map = {
            "input": 'mul_out',
            "linear_out": 'mlp_out'
        }

        down_linear_parallel_param = {
            "op_name": "down_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": down_linear_param,
            "enable_lcoc": True if is_prefill else False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(down_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, down_linear_tensor_map)


    def build_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """Build base MLP block computation graph: gateup + activation + down + residual."""
        mlp_res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                         op_name='mlp_res_add')
        setattr(graph, 'mlp_res_add', mlp_res_add)

        self.build_gateup_graph(graph)
        self.build_activation_graph(graph)
        self.build_down_graph(graph, is_prefill)

        graph.add_operation(graph.mlp_res_add, ['hidden_states', 'mlp_out'], ['layer_out'])


class BaseModelATB(torch.nn.Module):
    """
    Base class for Python graph model.

    Args:
        config (PretrainedConfig): Model configuration.
        weights (Weights): Model weights.
        model_prefix (str, optional): Prefix string of the model, default to "model".
        lm_head_prefix (str, optional): Prefix string of the language model head, default to "lm_head".
        is_parrllel (bool, optional): Whether the model is parallelized, default to False.
        is_fa (bool): Whether to use flash attention, default to False.
        backend (CommuicationBackend): Communication backend used in multi-card computation,
            default to `CommunicationBackedn.LCCL`.
    """
    def __init__(self, config: PretrainedConfig, weights: Weights, model_prefix: str = "model",
                 lm_head_prefix: str = "lm_head", is_parallel: bool = False, is_fa: bool = False,
                 backend: CommunicationBackend = CommunicationBackend.LCCL):
        super().__init__()

        # 配置信息
        self.config = config
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.model_prefix = model_prefix
        self.is_parallel = is_parallel
        self.weight_names = None
        self.is_fa = is_fa
        self.backend = backend
        # 模型结构
        self.embed_tokens = (TensorParallelEmbedding if self.is_parallel and not is_fa else TensorEmbedding)(
            prefix=f"{model_prefix}.embed_tokens", weights=weights
        )

    def get_weights(self, prefix: str) -> OrderedDict:
        """Get weights of the model."""
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, nn.ModuleList):
                for i, single_module in enumerate(module):
                    weights_dict.update(single_module.get_weights(f"{prefix}.{name}.{i}"))
            else:
                weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def build_word_embedding_graph(self, graph: atb._GraphOperation):
        """Build word embedding computation graph: initialize parameteros and build graph."""
        word_embedding_param = {
            "op_name": "word_embedding",
            "category": CommonOpBuilderType.WORD_EMBEDDING,
            "enable_parallel": self.is_parallel and not self.is_fa,
            "unpad_inputs": True,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
        }
        word_embedding_tensor_map = {
            "embedding_weights": f"{self.model_prefix}.embed_tokens.weight",
            "input_ids": "input_ids",
            "word_embedding_out": "hidden_states"
        }
        builder = CommonOpBuilderManager.get_builder(word_embedding_param)
        graph = builder.build(graph, word_embedding_tensor_map)

    def build_positional_embedding_graph(self, graph: atb._GraphOperation):
        """Build positional embedding computation graph: initialize parameters and build graph."""
        positional_embedding_param = {
            "op_name": "positional_embedding",
            "category": CommonOpBuilderType.POSITIONAL_EMBEDDING
        }
        positional_embedding_tensor_map = {
            "position_ids": "position_ids",
            "cos_table": "cos_table",
            "sin_table": "sin_table",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding"
        }
        builder = CommonOpBuilderManager.get_builder(positional_embedding_param)
        graph = builder.build(graph, positional_embedding_tensor_map)

    @abstractmethod
    def build_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """Abstract method for building graph."""
        pass
