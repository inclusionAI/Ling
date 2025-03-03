# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Qwen2-VL-VIT model."""

import json
import math
from collections import OrderedDict, defaultdict

import _libatb_torch as atb
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch_npu
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import (
    CommonOpBuilderType,
    NormType,
)
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo,
    CommunicationBackend,
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
)
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.utils import (
    add_start_docstrings,
)

_LINEAR_MODULE = "linear_module"
_INPUT = "input"
_LINEAR_OUT = "linear_out"
_LINEAR_PARAM = "linear_param"
_OP_NAME = "op_name"
_CATEGORY = "category"
_VIT = "VIT"
_PADDING_HEAD_DIM = 128


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self._seq_len_cached = 0
        self.scaling_factor = 1.0
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.freqs = None

    def update_freq(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached_total.device != device
                or self._cos_cached_total.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            t = t / self.scaling_factor
            # Don't do einsum, it converts fp32 to fp16 # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self.freqs = torch.outer(t, self.inv_freq.to(device=t.device))

    def get_freqs(self):
        return self.freqs


class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: int = 14,
            temporal_patch_size: int = 2,
            in_channels: int = 3,
            embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states).reshape(-1, self.embed_dim).contiguous()
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class LayerNormATB(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.layer_norm_eps = 1e-6
        self.prefix = prefix
        weight = weights.get_tensor(f"{prefix}.weight")
        bias = weights.get_tensor(f"{prefix}.bias")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight"] = self.weight.data
        weights_dict[f"{prefix}.bias"] = self.bias.data
        return weights_dict

    def build_graph(self, graph):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis": 2,
                "beginNormAxis": 2
            },
        }
        norm_op = atb._BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="norm",
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op, ["hidden_states", f"{self.prefix}.weight", f"{self.prefix}.bias"], [f"{self.prefix}_out"]
        )


class VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, weights, prefix, norm_prefix, backend):
        super().__init__()
        self.prefix = prefix
        self.norm_prefix = norm_prefix
        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.num_heads = config.num_heads
        self.hidden_size = config.embed_dim
        self.embed_dim = config.embed_dim
        self.head_dim_ori = self.embed_dim // self.num_heads
        self.head_dim = _PADDING_HEAD_DIM

        self.num_heads_pre_rank = (
                                          self.num_heads + self.tp_world_size - 1
                                  ) // self.tp_world_size
        setattr(config, 'quantize', None)
        self.quantize = config.quantize
        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{self.prefix}.qkv",
            weights=weights,
            bias=True,
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
        )

        self.proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{self.prefix}.proj",
            weights=weights,
            bias=True,
            gqa_size=self.head_dim_ori,
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        qkv_weights_dict = self.qkv.linear.get_weights(f"{prefix}.qkv")
        qkv_proj_weight = qkv_weights_dict[f"{prefix}.qkv.weight"]
        qkv_proj_bias = qkv_weights_dict[f"{prefix}.qkv.bias"]
        
        # padding head_dim from 80 to 128
        first_half = qkv_proj_weight.reshape(self.num_heads_pre_rank, 3, 80, self.embed_dim)[:, :, :40, :]
        second_half = qkv_proj_weight.reshape(self.num_heads_pre_rank, 3, 80, self.embed_dim)[:, :, 40:, :]
        first_half_padded = torch.nn.functional.pad(first_half, (0, 0, 0, 24))
        second_half_padded = torch.nn.functional.pad(second_half, (0, 0, 0, 24))
        qkv_proj_weight_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        qkv_proj_weight_final = qkv_proj_weight_padded.reshape(self.num_heads_pre_rank * 128 * 3, self.embed_dim)

        first_half = qkv_proj_bias.reshape(self.num_heads_pre_rank, 3, 80)[:, :, :40]
        second_half = qkv_proj_bias.reshape(self.num_heads_pre_rank, 3, 80)[:, :, 40:]
        first_half_padded = torch.nn.functional.pad(first_half, (0, 24))
        second_half_padded = torch.nn.functional.pad(second_half, (0, 24))
        qkv_proj_bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        qkv_proj_bias_final = qkv_proj_bias_padded.reshape(self.num_heads_pre_rank * 128 * 3)

        qkv_weights_dict[f"{prefix}.qkv.weight"] = qkv_proj_weight_final
        qkv_weights_dict[f"{prefix}.qkv.bias"] = qkv_proj_bias_final

        out_proj_weights_dict = self.proj.linear.get_weights(f"{prefix}.proj")
        out_proj_weight = out_proj_weights_dict[f"{prefix}.proj.weight"]
        soc_info = NPUSocInfo()
        if self.tp_world_size == 1 or soc_info.need_nz:
            out_proj_weight = torch.nn.functional.pad(
                    out_proj_weight.reshape(self.embed_dim, self.num_heads_pre_rank * 2, 40), (0, 24, 0, 0)
                ).reshape(self.embed_dim, self.num_heads_pre_rank * 128)
        elif self.tp_world_size > 1:
            first_half = out_proj_weight.reshape(self.num_heads_pre_rank, 80, self.embed_dim)[:, :40, :]
            second_half = out_proj_weight.reshape(self.num_heads_pre_rank, 80, self.embed_dim)[:, 40:, :]
            first_half_padded = torch.nn.functional.pad(first_half, (0, 0, 0, 24))
            second_half_padded = torch.nn.functional.pad(second_half, (0, 0, 0, 24))
            out_proj_weight_padded = torch.cat([first_half_padded, second_half_padded], dim=1)
            out_proj_weight = out_proj_weight_padded.reshape(self.num_heads_pre_rank * 128, self.embed_dim)

        out_proj_weights_dict[f"{prefix}.proj.weight"] = out_proj_weight
        weights_dict.update(qkv_weights_dict)
        weights_dict.update(out_proj_weights_dict)
        return weights_dict

    def build_qkv_graph(self, graph):
        input_key_list = [f"{self.norm_prefix}_out", f"{self.prefix}.qkv.weight", f"{self.prefix}.qkv.bias"]
        linear_out = ["qkv_linear_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="qkv" + "_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )
        split_op = atb._BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name="qkv" + "_Split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            ["qkv_linear_out"],
            ["q_split", "k_split", "v_split"],
        )

    def build_rope_graph(self, graph):
        rope_param = {
            _OP_NAME: "rope",
            "head_num": self.num_heads_pre_rank,
            "kv_head_num": self.num_heads_pre_rank,
            _CATEGORY: CommonOpBuilderType.ROPE,
            "is_fa": True,
            "atb_rope_param": {"rotaryCoeff": 2},
        }
        rope_tensor_map = {
            "q": "q_split",
            "k": "k_split",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding",
            "seq_len": "seq_len",
            "q_out": "q_split",
            "k_out": "k_split",
        }
        rope_builder = CommonOpBuilderManager.get_builder(rope_param)
        graph = rope_builder.build(graph, rope_tensor_map)

    def reshape_qkv(self, org_shape):
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]

    def reshape_out(self, org_shape):
        return [1, org_shape[0], self.num_heads_pre_rank * org_shape[2]]

    def build_attention_graph(self, graph):
        attention_op = atb._BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps({
                "headNum": self.num_heads_pre_rank,
                "kvHeadNum": self.num_heads_pre_rank,
                "qkScale": 1.0 / math.sqrt(self.head_dim_ori),
                "calcType": "PA_ENCODER"}),
            op_name="selfattention"
        )
        graph.add_reshape("q_split", "q_split_reshape", self.reshape_qkv)
        graph.add_reshape("k_split", "k_split_reshape", self.reshape_qkv)
        graph.add_reshape("v_split", "v_split_reshape", self.reshape_qkv)
        input_key_list = ["q_split_reshape", "k_split_reshape", "v_split_reshape", "seq_len"]
        output_key_list = ["attn_out"]
        graph.operations.append(attention_op)
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape("attn_out", "attn_out_reshape", self.reshape_out)

    def build_dense_graph(self, graph):
        dense_linear_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.proj.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        dense_linear_tensor_map = {
            "input": 'attn_out_reshape',
            "linear_out": 'dense_out'
        }
        dense_linear_parallel_param = {
            "op_name": "dense_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": dense_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(dense_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, dense_linear_tensor_map)

    def build_graph(self, graph):
        atten_res_add = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="atten_res_add",
        )
        setattr(graph, "atten_res_add", atten_res_add)
        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph)
        self.build_dense_graph(graph)
        graph.operations.append(graph.atten_res_add)
        graph.add_operation(
            graph.atten_res_add, ["hidden_states", "dense_out"], ["hidden_states"]
        )


class VisionMlp(nn.Module):
    def __init__(self, config, weights, prefix, norm_prefix, backend):
        super().__init__()
        self.norm_prefix = norm_prefix
        self.config = config
        self.weights = weights
        self.dtype = weights.dtype
        self.backend = backend

        self.prefix = prefix
        setattr(config, 'quantize', None)
        self.quantize = config.quantize
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.fc1 = TensorParallelColumnLinear.load(config,
                                                   prefix=f"{prefix}.fc1",
                                                   weights=weights,
                                                   bias=True, )
        self.fc2 = TensorParallelRowLinear.load(config,
                                                prefix=f"{prefix}.fc2",
                                                weights=weights,
                                                bias=True, )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.fc1.linear.get_weights(f"{prefix}.fc1"))
        weights_dict.update(self.fc2.linear.get_weights(f"{prefix}.fc2"))
        return weights_dict

    def build_fc1_graph(self, graph, input_tensor_name, output_tensor_name):
        input_key_list = [input_tensor_name, f"{self.prefix}.fc1.weight", f"{self.prefix}.fc1.bias"]
        linear_out = [output_tensor_name]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="fc1" + "_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )

    def build_activation_graph(self, graph, input_tensor_name, output_tensor_name):
        act = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_GELU'}),
            op_name="Activation",
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            [input_tensor_name],
            [output_tensor_name],
        )

    def build_fc2_graph(self, graph, input_tensor_name, output_tensor_name):
        fc2_param = {
            _OP_NAME: "fc2",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            _LINEAR_MODULE: self.fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        fc2_tensor_map = {
            _INPUT: input_tensor_name,
            _LINEAR_OUT: output_tensor_name,
        }

        fc2_parallel_param = {
            _OP_NAME: "fc2_parallel",
            _CATEGORY: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            _LINEAR_PARAM: fc2_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            fc2_parallel_param
        )
        graph = linear_parallel_builder.build(graph, fc2_tensor_map)

    def build_graph(self, graph):
        norm_out_name = f'{self.norm_prefix}_out'
        mlp_out_name = "mlp_out"
        mlp_res_add = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="mlp_res_add",
        )
        setattr(graph, "mlp_res_add", mlp_res_add)
        fc1_out_name = f'{norm_out_name}_fc1'
        act_out_name = f'{fc1_out_name}_act'
        self.build_fc1_graph(graph, norm_out_name, fc1_out_name)
        self.build_activation_graph(graph, fc1_out_name, act_out_name)
        self.build_fc2_graph(graph, act_out_name, mlp_out_name)

        graph.add_operation(
            graph.mlp_res_add, ["hidden_states", mlp_out_name], ["layer_out"]
        )


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config, layer_idx, weights, model_prefix, backend) -> None:
        super().__init__()
        prefix = f"{model_prefix}.{layer_idx}"
        self.layer_idx = layer_idx
        self.config = config
        self.weight_names = None
        self.layer_graph = None

        self.norm1 = LayerNormATB(config, weights, f"{prefix}.norm1")

        self.attn = VisionAttention(
            config=config,
            weights=weights,
            prefix=f"{prefix}.attn",
            norm_prefix=f"{prefix}.norm1",
            backend=backend,
        )

        self.norm2 = LayerNormATB(
            config, weights, f"{prefix}.norm2"
        )

        self.mlp = VisionMlp(
            config=config,
            weights=weights,
            prefix=f"{prefix}.mlp",
            norm_prefix=f"{prefix}.norm2",
            backend=backend,
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_in_tensor_names(self):
        default_input = ["hidden_states", "seq_len", "cos_embedding", "sin_embedding"]
        return default_input

    def build_graph(self, graph):
        self.layer_graph = AtbGraph(f"{_VIT}_layer_{self.layer_idx}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names
                  + self.get_in_tensor_names(),
            output=["layer_out"],
        )

        self.norm1.build_graph(self.layer_graph)
        self.attn.build_graph(self.layer_graph)
        self.norm2.build_graph(self.layer_graph)
        self.mlp.build_graph(self.layer_graph)
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        graph.add_operation(
            self.layer_graph,
            self.weight_names
            + self.get_in_tensor_names(),
            ["hidden_states"],
        )


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
)
class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2VLVisionEncoder(nn.Module):

    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.weights = weights
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.soc_info = NPUSocInfo()
        self.backend = (
            CommunicationBackend.HCCL
            if self.soc_info.need_nz
            else CommunicationBackend.LCCL
        )
        self.model_prefix = 'visual.blocks'
        layers = []
        for layer_idx in range(config.depth):
            layers.append(Qwen2VLVisionBlock(config, layer_idx, weights, self.model_prefix, self.backend))
        self.layers = nn.ModuleList(layers)

        self.graph = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()

    def prepare_inputs(self, hidden_states, seqlens, cos_vit_mrope, sin_vit_mrope):
        self.graph_inputs[_VIT].update({"hidden_states": hidden_states})
        self.graph_inputs[_VIT].update({"seq_len": seqlens})
        self.graph_inputs[_VIT].update({"cos_embedding": cos_vit_mrope})
        self.graph_inputs[_VIT].update({"sin_embedding": sin_vit_mrope})
        self.graph_param[_VIT]["seq_len"] = seqlens.cpu().to(torch.int32)
        self.graph_outputs[_VIT]['hidden_states'] = hidden_states

    def forward(
            self,
            hidden_states,
            seqlens,
            cos_vit_mrope,
            sin_vit_mrope
    ):

        self.prepare_inputs(hidden_states, seqlens, cos_vit_mrope, sin_vit_mrope)
        hidden_states = self.graph.forward(self.graph_inputs[_VIT], self.graph_outputs[_VIT], self.graph_param[_VIT])
        return hidden_states

    def get_in_tensor_names(self):
        return ['hidden_states', 'seq_len', "cos_embedding", "sin_embedding"]

    def get_out_tensor_names(self):
        return ['hidden_states']

    def get_weights(self):
        weights_dict = OrderedDict()
        layer_idx = 0
        for layer in self.layers:
            weights = layer.get_weights(f"{self.model_prefix}.{layer_idx}")
            weights_dict.update(weights)
            layer_idx += 1
        return weights_dict

    def build_graph(self):
        self.graph.add_input_output(input=list(self.weight.keys()) + self.get_in_tensor_names(),
                                    output=self.get_out_tensor_names())
        for layer in self.layers:
            layer.build_graph(self.graph)
        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("qwen2vl_vit_graph")
        self.build_graph()


class Qwen2VisionTransformerPretrainedModelATB(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionEncoder"]

    def __init__(self, config, weights, max_seq_len) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.weights = weights
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.encoder = Qwen2VLVisionEncoder(config, self.weights)
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )
        self.rotary_pos_emb.update_freq(weights.dtype, weights.device, max_seq_len)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        freqs = self.rotary_pos_emb.get_freqs()
        freqs_all = freqs[pos_ids].flatten(1)
        cos = freqs_all.cos()
        sin = freqs_all.sin()
        cos = torch.nn.functional.pad(cos, (0, 24))
        sin = torch.nn.functional.pad(sin, (0, 24))
        cos_vit_mrope = cos.repeat(1, 2).unsqueeze(0).to(self.weights.dtype)
        sin_vit_mrope = sin.repeat(1, 2).unsqueeze(0).to(self.weights.dtype)
        return cos_vit_mrope, sin_vit_mrope

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        cos_vit_mrope, sin_vit_mrope = self.rot_pos_emb(grid_thw)
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        hidden_states = torch_npu.npu_format_cast(hidden_states, 2)
        hidden_states = self.encoder(hidden_states.unsqueeze(0), seqlens.to(torch.int32), cos_vit_mrope, sin_vit_mrope)
        output = self.merger(hidden_states["hidden_states"])
        return output
