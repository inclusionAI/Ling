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
import json
from collections import OrderedDict

from torch import nn

import _libatb_torch as atb
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, OperationBackend
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import CommunicationBackend
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import AttnType
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.models.base.modeling_atb import BaseRMSNorm, BaseAttention, BaseMLP, BaseModelATB
from atb_llm.models.base.model_utils import LmHeadLinearInfo
from atb_llm.utils import OpBackend
from atb_llm.utils.layers import (
    KvCache,
)

QWEN_EMBEDDING_PARALLEL_THRESHOLD = 128256  # vocab size of qwen2


class QwenAttention(BaseAttention):
    def __init__(
        self,
        config,
        weights,
        prefix: str,
        norm_prefix: str,
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
        speculate_enable=False,
        bias=True
    ):
        super().__init__(config, weights, prefix, norm_prefix, is_fa, backend, bias)

        # 并行解码
        self.speculate_enable = speculate_enable
        # kv cache量化
        self.kv_quant = config.quantization_config.kv_quant_type
        self.kv_cache_quant = None

        if self.kv_quant is not None:
            self.kv_cache_quant = KvCache.load(prefix_k=f"{prefix}.k_proj",
                prefix_v=f"{prefix}.v_proj", weights=weights, backend=OpBackend.ATB)

    def get_weights(self, prefix):
        weights_dict = super().get_weights(prefix)
        if self.kv_quant is not None:
            weights_dict.update(self.kv_cache_quant.get_weights(f"{prefix}"))
        return weights_dict

    def build_attention_graph(self, graph, is_prefill):
        attention_param = {
            "op_name": "attention",
            "category": CommonOpBuilderType.ATTENTION,
            "is_prefill": is_prefill,
            "attn_type": AttnType.FLASH_ATTENTION if self.is_fa else AttnType.PAGED_ATTENTION,
            "head_size": self.head_size,
            "atb_reshape_and_cache_param": {},
            "operation_backend": OperationBackend.ATB,
            "enable_kv_quant": self.kv_quant is not None,
            "kv_quant_module": self.kv_cache_quant,
        }

        atb_attention_param = self._get_atb_attention_param(is_prefill)
        if not self.is_fa and not is_prefill and self.speculate_enable:
            atb_attention_param.update({
                    'maskType': 'MASK_TYPE_SPEC',
                    'calcType': 'CALC_TYPE_SPEC',
            })
        attention_param.update({"atb_attention_param": atb_attention_param})

        attention_tensor_map = self._get_attention_tensor_map()
        if not self.is_fa and self.speculate_enable:
            attention_tensor_map.update({"q_len": "q_len"})

        pa_attention_builder = CommonOpBuilderManager.get_builder(attention_param)
        graph = pa_attention_builder.build(graph, attention_tensor_map)

    def build_graph(self, graph, is_prefill):
        atten_res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                           op_name='atten_res_add')
        setattr(graph, 'atten_res_add', atten_res_add)

        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph, is_prefill)
        self.build_dense_graph(graph, is_prefill)

        graph.add_operation(graph.atten_res_add, ['hidden_states', 'dense_out'], ['hidden_states'])


class QwenLayer(nn.Module):
    def __init__(
        self,
        layer_id,
        config,
        weights,
        model_prefix: str = "model",
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
        speculate_enable: bool = False,
    ):
        super().__init__()

        # 配置信息
        prefix = f"{model_prefix}.layers.{layer_id}"
        self.layer_id = layer_id
        self.config = config
        tp_world_size = weights.process_group.size()
        self.is_reshape = config.vocab_size >= QWEN_EMBEDDING_PARALLEL_THRESHOLD and tp_world_size > 1 and not is_fa
        self.weight_names = None
        self.layer_graph = None
        self.is_fa = is_fa
        self.speculate_enable = speculate_enable

        # 模型结构
        self.self_attn = QwenAttention(
            config=config, weights=weights, prefix=f"{prefix}.self_attn", norm_prefix=f"{prefix}.input_layernorm", \
            is_fa=self.is_fa, backend=backend, speculate_enable=self.speculate_enable
        )

        self.mlp = BaseMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights,
            norm_prefix=f"{prefix}.post_attention_layernorm", backend=backend
        )

        self.input_layernorm = BaseRMSNorm(
            f"{prefix}.input_layernorm", config, weights, self.self_attn.linear_info
        )

        self.post_attention_layernorm = BaseRMSNorm(
            f"{prefix}.post_attention_layernorm", config, weights, self.mlp.linear_info
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_in_tensor_names(self, is_prefill):
        default_input = ['hidden_states', 'seq_len']
        if self.is_fa:
            default_input.extend(['token_offset', 'layer_id'])
        else:
            default_input.extend(['slots_mapping'])

        default_input.extend(['cos_embedding', 'sin_embedding'])
        if is_prefill or self.is_fa:
            default_input.extend(['attention_mask'])
        else:
            default_input.extend(['block_tables'])
            if self.speculate_enable:
                default_input.extend(['attention_mask', 'q_len'])
        return default_input

    def reshape_parallel(self, org_shape):
        if len(org_shape) == 3:
            if self.layer_id == 0:
                return [org_shape[0], org_shape[1] * org_shape[2]]
            else:
                return [org_shape[1], org_shape[0] * org_shape[2]]
        else:
            return org_shape

    def build_graph(self, graph, is_prefill):
        hidden_states = "hidden_states"
        self.layer_graph = AtbGraph(("prefill" if is_prefill else "decode") + f"_layer_{self.layer_id}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names + ["k_cache", "v_cache"] + self.get_in_tensor_names(is_prefill),
            output=["layer_out"])
        if self.is_reshape:
            self.layer_graph.add_reshape(hidden_states, hidden_states, self.reshape_parallel)
        self.input_layernorm.build_graph(self.layer_graph, is_prefill)
        self.self_attn.build_graph(self.layer_graph, is_prefill)
        self.post_attention_layernorm.build_graph(self.layer_graph, is_prefill)
        self.mlp.build_graph(self.layer_graph, is_prefill)
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        graph.add_operation(self.layer_graph, self.weight_names + \
        [f"layer_{self.layer_id}_k_cache", f"layer_{self.layer_id}_v_cache"] + self.get_in_tensor_names(
            is_prefill), [hidden_states])


class QwenModelATB(BaseModelATB):
    def __init__(
        self,
        config,
        weights,
        model_prefix: str = "model",
        lm_head_prefix: str = "lm_head",
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
        speculate_enable: bool = False
    ):
        is_parallel = config.vocab_size >= QWEN_EMBEDDING_PARALLEL_THRESHOLD
        super().__init__(config, weights, model_prefix, lm_head_prefix, is_parallel, is_fa, backend)

        self.layers = nn.ModuleList(
            [QwenLayer(layer_idx, config, weights, model_prefix, self.is_fa, self.backend, speculate_enable) \
             for layer_idx in range(config.num_hidden_layers)]
        )

        linear_info = LmHeadLinearInfo()
        linear_info.lm_head_name = lm_head_prefix
        self.norm = BaseRMSNorm(f"{model_prefix}.norm", config, weights, linear_info)

    def build_graph(self, graph, is_prefill):
        self.build_word_embedding_graph(graph)
        self.build_positional_embedding_graph(graph)

        for layer in self.layers:
            layer.build_graph(graph, is_prefill)

        self.norm.build_graph(graph, is_prefill)
