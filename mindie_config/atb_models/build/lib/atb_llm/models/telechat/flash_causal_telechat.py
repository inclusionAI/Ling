# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
"""PyTorch TELECHAT model."""

import json
import math
from typing import Optional, List, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
import torch.distributed
import torch_npu
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorEmbedding,
    PositionRotaryEmbedding,
    AttentionMask,
    TensorParallelHead,
    load_column_multi,
)
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import (
    AttnWrapper,
    MlpWrapper,
    WeightWrapper,
)
from atb_llm.utils.layers.linear import get_linear
from atb_llm.utils.dist import get_rank_table_file
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from .config_telechat import TelechatConfig


class RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class RMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class RMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = RMSNorm(prefix, weights, eps)
        self.anti = RMSNormBias(f"{prefix}.module", weights, eps)


class FlashTelechatAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config: TelechatConfig,
        weights,
    ):
        super().__init__()
        self.num_heads = config.n_head
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            dim=self.head_dim, base=10000.0, device="cpu"
        ).to(weights.device)
        self.softmax_scale = self.head_dim**-0.5

        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        linear_names = [f"{prefix}.query", f"{prefix}.key_value"]
        pack_name = f'{prefix}.w_pack'
        layer_prefix = ".".join(prefix.split(".")[:-1])
        norm_name = f"{layer_prefix}.input_layernorm"
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        if self.pack_type in [
            PackType.ALL_W8A16,
            PackType.ALL_W8A8SC,
        ]:
            self.w_pack = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.w_pack",
                weights=weights,
                bias=False
            )
            self.w_pack_ori = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.w_pack_ori",
                weights=weights,
                bias=False
            )
            self.dense = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.dense",
                weights=weights,
                bias=True,
                bias_pre_add=True
            )
        else:
            q_weight = weights.get_multi_weights_col(
                [f"{prefix}.query"], quantize=config.quantize, dim=0
            )
            kv_weight = weights.get_weights_col_packed_kv(
                f"{prefix}.key_value", config.quantize, self.hidden_size, self.head_dim
            )

            if isinstance(q_weight, torch.Tensor):
                weight = torch.cat([q_weight, kv_weight], dim=0)
            else:
                weight = [torch.cat([q, kv], dim=0) for q, kv in zip(q_weight, kv_weight)]
                weight[3] = q_weight[3]
                weight[4] = q_weight[4]

            self.w_pack = TensorParallelColumnLinear(
                get_linear(weight, bias=None, quantize=config.quantize)
            )
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)

            self.w_pack_ori = TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.query", f"{prefix}.key_value"],
                weights=weights,
                bias=False,
                dim=0,
            )

            self.dense = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.dense",
                weights=weights,
                bias=True,
                bias_pre_add=True,
            )
        self.prefix = prefix


class TelechatMLP(nn.Module):
    def __init__(self, prefix, config: TelechatConfig, weights):
        super().__init__()
        act = config.hidden_act
        
        def gelu_activation(x, act):
            approximate_mode = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
            return torch.nn.functional.gelu(x, approximate_mode)
        
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: gelu_activation(x, act)
        )
        
        linear_names = [f"{prefix}.gate_proj", f"{prefix}.up_proj"]
        pack_name = f"{prefix}.gate_up_proj"
        layer_prefix = ".".join(prefix.split(".")[:-1])
        norm_name = f"{layer_prefix}.post_attention_layernorm"
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        if self.pack_type in [
            PackType.ALL_FP,
            PackType.ALL_W8A8,
            PackType.ALL_W8A8_ANTI,
            PackType.ALL_W8A16,
        ]:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )
        elif self.pack_type in [PackType.ALL_W8A8SC]:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False
            )
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

        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=True,
            bias_pre_add=True,
        )

        try:
            self.intermediate_size = math.ceil(
                config.intermediate_size / weights.process_group.size()
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e


class TelechatBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        if config.n_layer == 38:
            if config.quantize == 'w8a8sc':
                prefix = f"model.h.{layer_id}"
            else:
                prefix = f"transformer.h.{layer_id}"
        else:
            if config.quantize == 'w8a8':
                prefix = f"transformer.h.{layer_id}"
            else:
                prefix = f"h.{layer_id}"

        self.self_attention = FlashTelechatAttention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights
        )
        self.mlp = TelechatMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        if self.self_attention.pack_type in [
            PackType.ALL_W8A8_ANTI,
            PackType.MIX_W8A8_ANTI,
        ]:
            self.input_layernorm = RMSNormWrapper(
                prefix=f"{prefix}.input_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.self_attention.pack_type in [PackType.ALL_W8A8, PackType.ALL_W8A8SC]:
            self.input_layernorm = RMSNormBias(
                prefix=f"{prefix}.input_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        else:
            self.input_layernorm = RMSNorm(
                prefix=f"{prefix}.input_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )

        if self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.post_attention_layernorm = RMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.ALL_W8A8SC]:
            self.post_attention_layernorm = RMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        else:
            self.post_attention_layernorm = RMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )


class FlashTelechatModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        if config.n_layer == 38:
            if config.quantize == 'w8a8sc':
                prefix = "model."
            else:    
                prefix = "transformer."
        else:
            if config.quantize == 'w8a8':
                prefix = "transformer."
            else:
                prefix = ""

        self.word_embeddings = TensorEmbedding(
            prefix=f"{prefix}word_embeddings", weights=weights
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                TelechatBlock(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        # Final Layer Norm
        self.ln_f = RMSNorm(
            prefix=f"{prefix}ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.gradient_checkpointing = False


class FlashTelechatForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.ascend_weight = None
        self.model = FlashTelechatModel(config, weights)
        
        
        if config.quantize == 'w8a8sc':
            prefix = "model."
        elif config.quantize in ['w8a8', 'w8a8s']:
            prefix = "transformer."
        else:
            prefix = ""

        
        if config.n_layer == 30:
            self.lm_head = load_column_multi(
                config,
                prefixes=[f"{prefix}word_embeddings"],
                weights=weights,
                head_size=1,
                lm_head=True,
                norm=self.config.vocab_size == 125696,
            )
        elif config.n_layer == 38:
            if config.quantize == 'w8a8sc':
                self.lm_head = TensorParallelHead.load_weight(
                    config,
                    prefix="lm_head", # lm_head 头
                    weights=weights,
                    is_norm=True,  # 不生效的配置
                )
            else:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=["lm_head"], # lm_head 头
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
        
        self.num_heads = config.n_head
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_key_value_heads = self.num_heads

        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
            self.num_key_value_heads = math.ceil(
                self.num_key_value_heads / weights.process_group.size()
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        self.ascend_atten_mask_fake = None
        self.ascend_atten_mask = None
        self.acl_operation_inputs = None
        self.init_ascend_operations(config)
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.ascend_rotary_embedding = PositionRotaryEmbedding.static(
            dim=self.head_size, base=10000.0, device="cpu"
        ).to(weights.device)
        self.place_holder = torch.tensor([1], dtype=torch.float16, device="npu")

        self.num_attention_heads = self.num_heads
        self.num_layers = config.num_hidden_layers
        self.is_prefill = True
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch(
            "TransdataOperation"
        )
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)

    def maybe_format_cast(self, tensor):
        if not self.soc_info.need_nz:  # transdata 会额外占资源
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def init_position_rotary_embedding(
        self, position_ids: torch.Tensor, max_seq_len: int
    ):
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype, position_ids.device, max_seq_len
        )
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config):
        self.quantize = config.quantize

        self.max_position_embeddings = config.max_position_embeddings

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(
            "telechat_DecoderModel"
        )
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(
            "telechat_DecoderModel"
        )

        self.acl_operation_inputs = []
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.ascend_atten_mask = AttentionMask.static(self.max_base_len)
        self.ascend_atten_mask_fake = self.ascend_atten_mask.get_attn_mask(
            1, dtype=torch.float16, device="npu"
        )

    def get_weights(self):
        attn_module_names = AttnWrapper(
            norm_name="input_layernorm",
            pack_name="w_pack",
            o_name="dense",
            sep_names=None,
            wrapper_name="self_attention"
        )
        mlp_module_names = MlpWrapper(
            norm_name="post_attention_layernorm",
            pack_name="gate_up_proj",
            sep_names=["gate_proj", "up_proj"],
            down_name="down_proj",
            wrapper_name="mlp"
        )
        weight_wrapper = WeightWrapper(
            self.soc_info, self.tp_rank, attn_module_names, mlp_module_names
        )
        weight_wrapper.register_embedding(self.model.word_embeddings)
        for i in range(self.num_layers):
            layer = self.model.h[i]
            weight_wrapper.register_layer(
                layer,
                self.quantize,
            )
            if self.soc_info.need_nz:
                del layer.self_attention
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        
        return weight_wrapper

    def init_ascend_weight(self):

        weight = self.get_weights()
        self.ascend_weight = weight.weights
        linear_type = weight.linear_type
        pack_quant_config = weight.pack_quant_type
        linear_transpose_types = weight.linear_transpose_types

        # 设置模型参数
        rank_table_file = get_rank_table_file()
        coder_param = {
            "normEps": self.config.layer_norm_epsilon,
            "enableAddNorm": False,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_heads,
            "isFA": False,
            "isBF16": False,
            "packQuantType": pack_quant_config,
            "linearQuantType": linear_type,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or rank_table_file else "lccl",
            "rankTableFile": rank_table_file,
            "hiddenSize": self.hidden_size,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "isUnpadInputs": True,
            "linearHasBias": [[False, True, False, True]] * self.config.num_hidden_layers
        }
        encoder_param = {
            **coder_param,
            "isPrefill": True,
            "supportLcoc": False if self.soc_info.need_nz else True,
        }
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_ascend_kvcache(self, kv_cache):
        kcache_id_exist = not self.ascend_kcache_id or self.ascend_kcache_id != id(
            kv_cache[0][0]
        )
        vcache_id_exist = not self.ascend_vcache_id or self.ascend_vcache_id != id(
            kv_cache[0][1]
        )
        if kcache_id_exist or vcache_id_exist:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            logger.debug(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.debug(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(
                f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}"
            )

    def prepare_inputs_for_ascend(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        self.is_prefill = is_prefill
        self.rotary_embedding.update_cos_sin_cache_total(
            torch.float32, self.device, max_seq_len
        )
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        if self.is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_base_len / 16) * 16
                atten_mask = self.ascend_atten_mask.get_attn_mask(
                    pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.ascend_atten_mask.get_attn_mask(
                    self.max_base_len,
                    kv_cache[0][0].dtype,
                    kv_cache[0][0].device,
                )
        else:
            atten_mask = self.ascend_atten_mask_fake
        if self.is_prefill and lm_head_indices is None:  # prefill
            lm_head_indices = torch.tensor(
                range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device
            )

        self.acl_operation_inputs = [
            input_ids,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.place_holder,
            self.place_holder,
            self.place_holder,
            input_lengths.to(torch.int32),
            (
                lm_head_indices.to(torch.int64)
                if self.is_prefill
                else self.lm_head_indices_fake
            ),
        ]
        for ind, item in enumerate(self.acl_operation_inputs):
            logger.debug(f"{ind} {item.device=}")
        return self.acl_operation_inputs

    def execute_ascend_operator(
        self,
        acl_inputs,
        acl_param,
        is_prefill
    ):
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,  # input id, 拉平的
        position_ids: torch.Tensor,  #
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
        block_tables: torch.Tensor,  # 每个requests 所有的block tables
        slots: torch.Tensor,  # 每个requests 所有的slots
        input_lengths: torch.Tensor,  # 每个 request的k/v长度
        max_seq_len: int,  # 最长的request长度
        lm_head_indices: Optional[
            torch.Tensor
        ] = None,  # prefill阶段使用，取的生成token的偏移
        **kwargs,
    ) -> torch.Tensor:
        self.is_prefill = is_prefill
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)
        acl_inputs = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices
        )
        acl_param = json.dumps({"seqLen": input_lengths.tolist()})
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits
