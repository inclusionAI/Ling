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
from transformers.modeling_utils import PretrainedConfig
from atb_llm.utils import OpBackend
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    load_column_multi,
    KvCache,
    FA3,
    ReduceQuant,
    RMSNorm, RMSNormBias, RMSNormWrapper, RMSNormAntiOutlierWrapper
)
from atb_llm.utils.quantize.pack_type import get_pack_type, ALL_PACK_LIST, PackType
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.weights import Weights


def get_suffix(tensor_name: str) -> str:
    """Get the suffix of a tensor name."""
    return tensor_name.split(".")[-1]


class FlashAttention(torch.nn.Module):
    """
    Implementation of Flash Attention block.

    Args:
        prefix (str): The prefix of the layer.
        config (PretrainedConfig): The configuration of the model.
        weights (Weights): The weights of the model.
    """
    def __init__(self, prefix: str, config: PretrainedConfig, weights: Weights, **kwargs):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.weights = weights

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads

        self.qkv_names = [f'{self.prefix}.q_proj', f'{self.prefix}.k_proj', f'{self.prefix}.v_proj']
        self.qkv_bias = False
        self.dense_name = f'{self.prefix}.o_proj'
        self.dense_bias = False
        self.pack_name = f'{self.prefix}.query_key_value'
        layer_prefix = '.'.join(self.prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.input_layernorm'

        self.bias_pre_add = False

        self.pack_type = None
        self.kv_cache_quant = None
        self.fa3 = None
        self.reduce_quant = None

    def load_weights(self, **kwargs):
        """Load weights."""
        self.pack_type = get_pack_type(self.weights, self.qkv_names, self.norm_name, self.pack_name)

        if self.config.quantization_config.kv_quant_type is not None:
            if len(self.qkv_names) >= 3:
                k_name = self.qkv_names[1]
                v_name = self.qkv_names[2]
            else:
                k_name = f"{self.qkv_names[0]}.k_proj"
                v_name = f"{self.qkv_names[0]}.v_proj"
            self.kv_cache_quant = KvCache.load(prefix_k=k_name,
                prefix_v=v_name, weights=self.weights,
                backend=kwargs.get("attn_decode_backend", OpBackend.ATB))

        if self.config.quantization_config.fa_quant_type is not None:
            self.fa3 = FA3.load(
                prefix_q=f"{self.prefix}.fa_q", prefix_k=f"{self.prefix}.fa_k", prefix_v=f"{self.prefix}.fa_v",
                weights=self.weights, head_size=self.head_size)

        if self.config.quantization_config.reduce_quant_type is not None:
            self.reduce_quant = ReduceQuant.load(prefix=self.dense_name, weights=self.weights)

        self.load_qkv_weights(**kwargs)
        self.load_dense_weights(**kwargs)

    def load_qkv_weights(self, **kwargs):
        """Load qkv weights."""
        if self.pack_type in ALL_PACK_LIST and self.weights.quantize == QuantType.W8A8SC:
            query_key_value_linear = TensorParallelColumnLinear.load(
                self.config,
                prefix=self.pack_name,
                weights=self.weights,
                bias=self.qkv_bias,
            )
            setattr(self, get_suffix(self.pack_name), query_key_value_linear)
        elif len(self.qkv_names) == 1:
            query_key_value_linear = TensorParallelColumnLinear.load_qkv(
                self.config,
                prefix=self.qkv_names[0],
                weights=self.weights,
                bias=self.qkv_bias,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )
            setattr(self, get_suffix(self.pack_name), query_key_value_linear)
        elif self.pack_type in ALL_PACK_LIST and self.weights.quantize != QuantType.W8A8SC:
            query_key_value_linear = load_column_multi(
                self.config,
                prefixes=self.qkv_names,
                weights=self.weights,
                bias=self.qkv_bias,
                head_size=self.head_size,
            )
            setattr(self, get_suffix(self.pack_name), query_key_value_linear)
        else:
            for name in self.qkv_names:
                linear = TensorParallelColumnLinear.load(
                    self.config,
                    prefix=name,
                    weights=self.weights,
                    bias=self.qkv_bias,
                )
                setattr(self, get_suffix(name), linear)

    def load_dense_weights(self, **kwargs):
        """Load dense weights."""
        dense_linear = TensorParallelRowLinear.load(
            self.config,
            prefix=self.dense_name,
            weights=self.weights,
            bias=self.dense_bias,
            gqa_size=self.head_size,
            bias_pre_add=self.bias_pre_add
        )
        setattr(self, get_suffix(self.dense_name), dense_linear)


class MLP(torch.nn.Module):
    """
    Implementation of MLP block.

    Args:
        prefix (str): Prefix of the MLP block.
        config (PretrainedConfig): Configuration of the MLP block.
        weights (Weights): Weights of the MLP block.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, prefix: str, config: PretrainedConfig, weights: Weights, **kwargs):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.weights = weights

        self.gate_up_names = [f'{self.prefix}.gate_proj', f'{self.prefix}.up_proj']
        self.gate_up_bias = False
        self.down_name = f'{self.prefix}.down_proj'
        self.down_bias = False
        self.pack_name = f'{self.prefix}.gate_up_proj'
        layer_prefix = '.'.join(self.prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.up_weight_only = False

        self.bias_pre_add = False

        self.pack_type = None
        self.reduce_quant = None

    def load_weights(self, **kwargs):
        """Load weights."""
        self.pack_type = get_pack_type(
            self.weights, self.gate_up_names, self.norm_name, self.pack_name)

        if self.config.quantization_config.reduce_quant_type is not None:
            self.reduce_quant = ReduceQuant.load(prefix=self.down_name, weights=self.weights)
        self.load_gate_up_weights(**kwargs)
        self.load_down_weights(**kwargs)

    def load_gate_up_weights(self, **kwargs):
        """Load gate up weights."""
        if self.pack_type in ALL_PACK_LIST and self.weights.quantize == QuantType.W8A8SC:
            gate_up_linear = TensorParallelColumnLinear.load(
                self.config,
                prefix=self.pack_name,
                weights=self.weights,
                bias=self.gate_up_bias,
            )
            setattr(self, get_suffix(self.pack_name), gate_up_linear)
        elif self.up_weight_only:
            up_linear = TensorParallelColumnLinear.load(
                self.config,
                prefix=self.gate_up_names[0],
                weights=self.weights,
                bias=self.gate_up_bias,
            )
            setattr(self, get_suffix(self.gate_up_names[0]), up_linear)
        elif len(self.gate_up_names) == 1:
            gate_up_linear = TensorParallelColumnLinear.load_gate_up(
                self.config,
                prefix=self.gate_up_names[0],
                weights=self.weights,
                bias=self.gate_up_bias,
            )
            setattr(self, get_suffix(self.pack_name), gate_up_linear)
        elif self.pack_type in ALL_PACK_LIST:
            gate_up_linear = load_column_multi(
                    self.config,
                    prefixes=self.gate_up_names,
                    weights=self.weights,
                    head_size=1,
                    bias=self.gate_up_bias,
                )
            setattr(self, get_suffix(self.pack_name), gate_up_linear)
        else:
            for name in self.gate_up_names:
                linear = TensorParallelColumnLinear.load(
                    self.config,
                    prefix=name,
                    weights=self.weights,
                    bias=self.gate_up_bias,
                )
                setattr(self, get_suffix(name), linear)

    def load_down_weights(self, **kwargs):
        """Load down weights."""
        down_linear = TensorParallelRowLinear.load(
            self.config,
            prefix=self.down_name,
            weights=self.weights,
            bias=self.down_bias,
            bias_pre_add=self.bias_pre_add
        )
        setattr(self, get_suffix(self.down_name), down_linear)


class FlashLayer(torch.nn.Module):
    """
    Implementation of Flash Attention Layer.

    Args:
        layer_id (int): ID of the layer.
        config (PretrainedConfig): Configuration of the Flash Attention layer.
        weights (Weights): Weights of the Flash Attention layer.
        model_prefix (str, optional): Prefix of the model. Defaults to "model".
        **kwargs: Additional keyword arguments.

    """
    def __init__(self, layer_id: int, config: PretrainedConfig, weights: Weights,
                 model_prefix: str = "model", **kwargs):
        super().__init__()
        self.config = config
        self.weights = weights

        self.prefix = f"{model_prefix}.layers.{layer_id}"
        self.attn_name = "self_attn"
        self.mlp_name = "mlp"
        self.norm_bias = False
        self.norm_eps = 1e-5

    def load_weights(self, **kwargs):
        """Load weights for layer."""
        if self.norm_bias:
            self.norm_eps = getattr(self.config, "layer_norm_epsilon", 1e-5)
        else:
            self.norm_eps = getattr(self.config, "rms_norm_eps", 1e-5)
        self.load_input_layernorm_weight(**kwargs)
        self.load_post_attention_layernorm_weight(**kwargs)

    def load_input_layernorm_weight(self, **kwargs):
        """Load weights for input layernorm."""
        attn_module = getattr(self, self.attn_name)
        if attn_module.pack_type in [
            PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.ALL_W8A8_DYNAMIC]:
            norm_cls = RMSNormBias if self.norm_bias else RMSNorm
            input_layernorm = norm_cls(
                prefix=attn_module.norm_name, weights=self.weights, eps=self.norm_eps
            )
        elif attn_module.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            input_layernorm = RMSNormWrapper(
                prefix=attn_module.norm_name, weights=self.weights, eps=self.norm_eps
            )
        elif attn_module.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            input_layernorm = RMSNormAntiOutlierWrapper(
                prefix=attn_module.norm_name, weights=self.weights, eps=self.norm_eps
            )
        elif attn_module.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            input_layernorm = RMSNormBias(
                prefix=attn_module.norm_name, weights=self.weights, eps=self.norm_eps
            )
        else:
            logger.error("error: self attention pack type not supported", 
                         ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise AssertionError(f'self_attn.pack_type: {attn_module.pack_type} not supported')
        setattr(self, get_suffix(attn_module.norm_name), input_layernorm)

    def load_post_attention_layernorm_weight(self, **kwargs):
        """Load post attention layernorm weight."""
        mlp_module = getattr(self, self.mlp_name)
        if mlp_module.pack_type in [
            PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.ALL_W8A8_DYNAMIC]:
            norm_cls = RMSNormBias if self.norm_bias else RMSNorm
            post_attention_layernorm = norm_cls(
                prefix=mlp_module.norm_name,
                weights=self.weights,
                eps=self.norm_eps,
            )
        elif mlp_module.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            post_attention_layernorm = RMSNormWrapper(
                prefix=mlp_module.norm_name,
                weights=self.weights, eps=self.norm_eps
            )
        elif mlp_module.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            post_attention_layernorm = RMSNormAntiOutlierWrapper(
                prefix=mlp_module.norm_name,
                weights=self.weights, eps=self.norm_eps
            )
        elif mlp_module.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            post_attention_layernorm = RMSNormBias(
                prefix=mlp_module.norm_name,
                weights=self.weights,
                eps=self.norm_eps,
            )
        else:
            logger.error("error: mlp pack type not supported", 
                         ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise AssertionError(f'mlp.pack_type: {mlp_module.pack_type} not supported')
        setattr(self, get_suffix(mlp_module.norm_name), post_attention_layernorm)
