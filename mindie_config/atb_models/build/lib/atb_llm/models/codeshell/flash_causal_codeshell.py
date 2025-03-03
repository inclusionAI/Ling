# Copyright 2022 EleutherAI and the HuggingFace Inc. team
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
from typing import Optional, List, Tuple
import torch

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi, TensorHead
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from .config_codeshell import CodeshellConfig
from .modeling_codeshell import GPTTransformer


class CodeShellRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        if base == 0 or dim == 0:
            msg = 'base or dim can not be zero!'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        super().__init__()

        self.max_seq_len_cached = None
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype).squeeze(0).squeeze(0),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype).squeeze(0).squeeze(0),
        )


class TensorParallelColumnEmbedding(torch.nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=1)

        self.process_group = weights.process_group
        self.reduce = reduce

        self.weight = torch.nn.Parameter(weight)


class FlashCodeshellForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.config = config
        self.dtype = weights.dtype
        
        self.embedding_prefix = "transformer.wte"  # transformer.wte: name of embedding weight
        self.embedding = TensorParallelColumnEmbedding(
            prefix=self.embedding_prefix, weights=weights
        )
        self.encoder = GPTTransformer(config, weights)
        if config.quantize == "w8a8sc":
            self.output_layer = TensorHead.load_weight(
                config,
                prefix=self.embedding_prefix,
                weights=weights,
                is_norm=False
            )
        else:
            self.output_layer = load_column_multi(
                config,
                prefixes=[self.embedding_prefix],
                weights=weights,
                head_size=1,
                lm_head=True
            )

        self.gradient_checkpointing = False
        self.soc_info = NPUSocInfo()

        self.device = weights.device
        # for ascend init
        self.placeholder = torch.zeros(1, dtype=config.torch_dtype, device="npu")
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        self.seq_length = None
        self.init_ascend_operations(config)
        self.ascend_weight = None
        self.kv_head_num = max(config.multi_query_group_num // self.tp_world_size, 1)
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = CodeShellRotaryEmbedding(rotary_dim, config.max_position_embeddings,
                                                       device=weights.device)
        self.cos_embed, self.sin_embed = self.rotary_pos_emb.forward(self.placeholder, config.max_position_embeddings)

        self.acl_param_encoder = None
        self.acl_param_decoder = None
        if self.soc_info.need_nz:
            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)

    def init_ascend_operations(self, config: CodeshellConfig):
        self.quantize = config.quantize
        self.seq_length = config.max_position_embeddings

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("codeshell_7b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("codeshell_7b_DecoderModel")

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="cpu").to(self.device)

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        coder_param = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "isUnpadInputs": True,
            "isEmbeddingParallel": True,
            "enableAddNorm": False,
            "enableSwiGLU": False,
            "normType": NormType.LAYER_NORM,
            "normEps": self.config.layer_norm_epsilon,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": max(self.config.multi_query_group_num // self.tp_world_size, 1),
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": True,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "tokenOffset": [0],
            "seqLen": [1],
            "weightQuantType": self.config.quantize if self.config.quantize else "",
            "linearTransposeType": linear_transpose_types,
            "linearHasBias": [[True, True, True, True]] * self.config.num_hidden_layers,
            "lmHeadTransposeType": self.output_layer.linear.trans_flag,
        }
        encoder_param = {**coder_param, "isPrefill": True}
        decoder_param = {**coder_param, "isPrefill": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

        self.lm_head_indices_fake = self.lm_head_indices_fake.to(self.device)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=None,
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='c_fc',
            sep_names=None,
            down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.embedding)
        for i in range(self.num_layers):
            layer = self.encoder.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.attn
                del layer.ln_2
                del layer.mlp
        weight_wrapper.register_model_norm(self.encoder.final_layernorm)
        word_embeddings_linear = type('nnLinear', (object,), {'linear': self.embedding})()
        weight_wrapper.register_model_lmhead(word_embeddings_linear)
        return weight_wrapper

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        self.init_position_rotary_embedding(position_ids, max_seq_len)
        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype, kv_cache[0][0].device)
            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if self.dtype == torch.bfloat16:
                atten_mask = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64),
            ]

            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            if self.dtype == torch.bfloat16:
                atten_mask = torch.zeros(input_lengths.size(0),
                                        self.num_attention_heads,
                                        1, input_lengths.max().item(),
                                        dtype=self.dtype,
                                        device=input_ids.device)
            else:
                atten_mask = self.attn_mask_fake

            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                self.lm_head_indices_fake,
            ]

            return self.acl_decoder_operation_inputs, self.acl_param
