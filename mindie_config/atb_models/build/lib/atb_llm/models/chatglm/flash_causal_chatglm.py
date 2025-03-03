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
import math
from typing import Optional, List, Tuple

import torch
from atb_llm.utils.layers import TensorEmbedding, load_column_multi, TensorHead
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.layers.norm.fast_layer_norm import NormType
from .config_chatglm import ChatglmConfig
from .modeling_chatglm import GLMTransformer


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None, version=None):
        super().__init__()
        inv_freq = 1.0 / \
                   (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio
        self.version = version

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        theta = 1.0 / \
                (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / self.rope_ratio
        if self.version == 'v3_6b' or self.version == 'v4_9b':
            base = base * self.rope_ratio
            theta = 1.0 / \
                    (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

            # Create position indexes `[0, 1, ..., seq_len - 1]`
            seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        emb = torch.stack((idx_theta, idx_theta), dim=-1)
        rope_cos = torch.cos(emb)
        rope_sin = torch.sin(emb)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            if dtype == torch.bfloat16:
                rope_cos = rope_cos.bfloat16()
                rope_sin = rope_sin.bfloat16()
            else:
                rope_cos = rope_cos.half()
                rope_sin = rope_sin.half()

        return rope_cos, rope_sin

    def forward(self, max_seq_len):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config, weights):
        super(Embedding, self).__init__()

        if config.quantize == "w8a8sc":
            self.word_embeddings = TensorEmbedding(
                    prefix="embedding.word_embeddings", weights=weights
                )
        else:
            self.word_embeddings = TensorEmbedding(
                    prefix="transformer.embedding.word_embeddings", weights=weights
                )
        

class FlashChatglmForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.config = config
        self.dtype = weights.dtype
        try:
            self.version = self._get_version(config._name_or_path)
        except KeyError as e:
            raise e

        self.embedding = Embedding(config, weights)
        self.encoder = GLMTransformer(config, weights)
        if config.quantize == "w8a8sc":
            self.output_layer = TensorHead.load_weight(
                config,
                prefix="output_layer",
                weights=weights,
                is_norm=False
            )
        else:
            self.output_layer = load_column_multi(
                config,
                prefixes=["transformer.output_layer"],
                weights=weights,
                head_size=1,
                lm_head=True
            )

        self.gradient_checkpointing = False
        self.device = weights.device
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=config.rope_ratio,
                                              original_impl=config.original_rope, device=weights.device,
                                              dtype=config.torch_dtype, version=self.version)
        self.cos_embed, self.sin_embed = self.rotary_pos_emb.forward(config.max_position_embeddings)
        self.skip_word_embedding = False
        
        self.acl_param_encoder = None
        self.acl_param_decoder = None
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []

        self.ascend_weight = []
        self.placeholder = torch.zeros(1, dtype=config.torch_dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="cpu").to(self.device)

        self.soc_info = NPUSocInfo()
        if self.soc_info.need_nz:
            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)

        self.decoder_slots = None
        self.block_tables_global = None
        self.wins_global = None
        self.razor_offset = None
        self.in_ra_seqlens = None
        self.pffset_index = None
        self.decode_pffset_index = None
        self.in_reshape_seqlen = None
        self.block_nums_list = None

    @staticmethod
    def _get_version(name_or_path):
        if 'chatglm2' in name_or_path.lower() or 'codegeex2' in name_or_path.lower():
            version = 'v2_6b'
        elif 'chatglm3' in name_or_path.lower():
            version = 'v3_6b'
        elif 'glm-4' in name_or_path.lower():
            version = 'v4_9b'
        else:
            msg = ("Currently only chatglm2_6b, chatglm3_6b, codegeex2_6b, glm-4-9b are supported. "
                    "If it is the above model, "
                    "please check whether the content of the _name_or_path field in config.json is standardized")
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(msg)

        return version
    
    def init_ascend_operations(self, config: ChatglmConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("chatglm_ChatglmDecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("chatglm_ChatglmDecoderModel")

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        coder_param = {
            "enableAddNorm": False,
            "isUnpadInputs": True,
            "normType": NormType.RMS_NORM,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "skipWordEmbedding": False,
            "isEmbeddingParallel": False,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "normEps": self.config.layernorm_epsilon,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": max(self.config.multi_query_group_num // self.tp_world_size, 1),
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "isLmHeadParallel": True,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "weightQuantType": self.config.quantize if self.config.quantize else "",
            "quantGroupSize": self.config.quantization_config.group_size,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.output_layer.linear.trans_flag,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "enableCompressHead": self.compress_head_enable,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers
        }
        encoder_param = {
            **coder_param, "isPrefill": True,
            "enableSpeculate": False,
            "skipWordEmbedding": self.skip_word_embedding
        }
        decoder_param = {**coder_param, "isPrefill": False, "enableSpeculate": self.speculate_enable}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

        self.lm_head_indices_fake = self.lm_head_indices_fake.to(self.device)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attention',
            pack_name='query_key_value',
            sep_names=None,
            o_name='dense'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='dense_h_to_4h',
            sep_names=None,
            down_name='dense_4h_to_h'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.embedding.word_embeddings)
        for i in range(self.config.num_hidden_layers):
            layer = self.encoder.layers[i]
            weight_wrapper.register_layer(layer, self.config.quantize)
            if self.soc_info.need_nz:
                del layer.self_attention
                del layer.post_attention_layernorm
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.encoder.final_layernorm)
        weight_wrapper.register_model_lmhead(self.output_layer)
        return weight_wrapper

    def construct_ra_input(self,
                           input_lengths: torch.Tensor,
                           kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                           max_seq_len: int,
                           max_out_len: int):
        layer_nums = self.config.num_hidden_layers
        head_nums = self.num_key_value_heads
        batch_size = input_lengths.shape[0]
        block_size = kv_cache[0][0].shape[1]

        block_nums_list = [i[0].shape[0] * block_size for i in kv_cache]
        block_nums = int(max(block_nums_list) / block_size)
        self.block_nums_list = block_nums_list

        npu = "npu"

        # 1.根据离线校准获得的 head_dict 构造 windows
        wins = torch.zeros((batch_size, head_nums), dtype=torch.int32).npu()
        wins_keep = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32).npu()
        wins_drop = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32).npu()
        pffset_index = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32).npu()

        head_dict = {
            'prefix_matching': {
                19: [0, 1],
                20: [2],
                21: [0, 1, 3],
                22: [3],
                23: [0, 1, 3],
                24: [1, 2, 3],
                25: [2, 3],
                26: [2, 3],
                28: [0, 1, 2, 3],
                29: [0, 1, 2, 3],
                31: [2, 3],
                32: [0, 1, 2, 3],
                33: [0, 1, 2, 3],
                34: [0, 1, 2, 3],
            },
            'copying': {5: [2], 6: [3], 9: [0], 10: [0], 12: [2], 13: [1], 14: [2], 15: [2], 16: [2]}
        }
        inductive_head = head_dict["prefix_matching"]
        copying_head = head_dict["copying"]

        for batch_idx in range(batch_size):
            first_sink = 4
            last_sink = max(4000, input_lengths[batch_idx] // 5)

            if input_lengths[batch_idx] - first_sink - last_sink - 1 <= 0:  # 不需要压缩
                wins[batch_idx][:] = 0
            else:  # 需要压缩
                wins[batch_idx][:] = input_lengths[batch_idx] - first_sink - last_sink

            kv_tp_size = min(self.tp_world_size, self.config.multi_query_group_num)
            for layer_idx in range(layer_nums):
                for head_idx in range(head_nums):
                    cur_head_idx = head_idx + self.tp_rank * kv_tp_size // self.tp_world_size * head_nums
                    is_inductive_head = layer_idx in inductive_head and cur_head_idx in inductive_head[layer_idx]
                    is_copying_head = layer_idx in copying_head and cur_head_idx in copying_head[layer_idx]
                    # 不需要压缩的head
                    if (is_inductive_head or is_copying_head) or \
                        (input_lengths[batch_idx] - first_sink - last_sink - 1 <= 0):
                        wins_drop[batch_idx][layer_idx][head_idx] = 0
                        wins_keep[batch_idx][layer_idx][head_idx] = input_lengths[batch_idx]
                        pffset_index[batch_idx][layer_idx][head_idx] = -1
                    # 需要压缩的head
                    else:
                        wins_drop[batch_idx][layer_idx][head_idx] = \
                            input_lengths[batch_idx] - first_sink - last_sink
                        wins_keep[batch_idx][layer_idx][head_idx] = first_sink + 1 + last_sink
                        pffset_index[batch_idx][layer_idx][head_idx] = first_sink

        # 2.重新定义 block_tables
        if block_size != 0:
            max_need_blocks = math.ceil((max_seq_len + max_out_len) / block_size)
        else:
            max_need_blocks = 0
        block_tables = torch.zeros((batch_size, layer_nums, head_nums, max_need_blocks),
                                   dtype=torch.int32, device=npu)
        cur_need_blocks = torch.ceil((wins_keep.float() + max_out_len) / block_size).to(torch.int32)
        block_indices = (torch.arange(max_need_blocks, dtype=torch.int32, device=npu).
                         expand(batch_size, layer_nums, head_nums, max_need_blocks))
        global_offsets = torch.cumsum(cur_need_blocks, dim=-1, dtype=torch.int32) - cur_need_blocks
        valid_mask = block_indices < cur_need_blocks.unsqueeze(-1)
        broadcasted_block_indices = block_indices + global_offsets.unsqueeze(-1)
        valid_mask_indices = valid_mask.nonzero(as_tuple=True)
        block_tables[valid_mask_indices] = broadcasted_block_indices[valid_mask]

        # 3.重新定义 slots
        self.decoder_slots = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32, device=npu)
        offsets = (block_tables[:, :, :, 0] * block_size).to(torch.int32)
        seq_lens = wins_keep
        slots = offsets
        self.decoder_slots = offsets + seq_lens - 1

        # 4.定义 PageAttention 所需输入 ra_offset
        ra_offset = torch.zeros((layer_nums, block_nums * block_size), dtype=torch.float32, device=npu)
        mask = wins_drop > 0
        log_wins_drop = torch.log(wins_drop)
        valid_offsets = offsets + first_sink
        layer_indices = (torch.arange(layer_nums, dtype=torch.int32, device=npu).unsqueeze(0).unsqueeze(2).
                         expand(batch_size, layer_nums, head_nums))
        valid_offsets_flat = valid_offsets[mask]
        layer_indices_flat = layer_indices[mask]
        ra_offset.index_put_((layer_indices_flat, valid_offsets_flat), log_wins_drop[mask], accumulate=False)

        # reshape 成需要的维度
        in_ra_seqlens = wins_keep.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        block_tables = block_tables.transpose(0, 1).reshape(layer_nums, batch_size * head_nums, max_need_blocks)
        slots = slots.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        pffset_index = pffset_index.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        ra_offset = ra_offset.reshape(layer_nums, block_nums, block_size)

        self.decoder_slots = self.decoder_slots.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        self.block_tables_global = block_tables

        self.wins_global = wins.reshape(batch_size * head_nums)
        self.razor_offset = ra_offset
        self.in_ra_seqlens = in_ra_seqlens
        self.pffset_index = pffset_index

        self.decode_pffset_index = torch.full((layer_nums, batch_size * head_nums), -1, dtype=torch.int32, device=npu)

        return block_tables, slots

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
        if is_prefill:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]

            self.acl_param_encoder = json.dumps({
                "seqLen": input_lengths.tolist(), # seqLen: 序列长度
            })

            if self.compress_head_enable:
                max_out_len = kwargs.get('max_out_len', 256)
                block_tables, slots = self.construct_ra_input(input_lengths, kv_cache, max_seq_len, max_out_len)
                block_tables = block_tables.reshape(-1, block_tables.shape[-1])
                slots = slots.reshape(-1)

                self.acl_param_encoder = json.dumps({
                    "seqLen": input_lengths.tolist(),  # seqLen: 序列长度
                    "blockNumsList": self.block_nums_list
                })

            self.acl_encoder_operation_inputs = [
                input_ids,
                position_ids,
                self.cos_embed,
                self.sin_embed,
                atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64)
            ]
            if self.compress_head_enable:
                self.in_reshape_seqlen = input_lengths.to(torch.int32)
                self.acl_encoder_operation_inputs.append(self.wins_global)
                self.acl_encoder_operation_inputs.append(self.in_ra_seqlens.reshape(-1))
                self.acl_encoder_operation_inputs.append(self.pffset_index.reshape(-1))
                self.acl_encoder_operation_inputs.append(self.razor_offset.reshape(-1, self.razor_offset.shape[-1]))
                self.acl_encoder_operation_inputs.append(self.in_reshape_seqlen)

            return self.acl_encoder_operation_inputs, self.acl_param_encoder
        else:
            q_lens = kwargs.get('q_lens', [])
            spec_mask = kwargs.get('spec_mask', None)
            self.acl_param_decoder = json.dumps({
                "seqLen": input_lengths.tolist(), # seqLen: 序列长度
                "qLen": q_lens
            })
            if self.speculate_enable and self.soc_info.need_nz:
                spec_mask = self.transdata_operation.execute([spec_mask])[0]
            
            if self.dtype == torch.bfloat16:
                self.attn_mask_fake = torch.zeros(input_lengths.size(0),
                                         self.num_attention_heads,
                                         1,
                                         input_lengths.max(),
                                         dtype=self.dtype,
                                         device=input_ids.device)
            atten_mask = spec_mask if self.speculate_enable else self.attn_mask_fake

            if self.compress_head_enable:
                self.in_ra_seqlens = self.in_ra_seqlens + 1
                self.decoder_slots = self.decoder_slots + 1
                slots = self.decoder_slots
                block_tables = self.block_tables_global

                block_tables = block_tables.reshape(-1, block_tables.shape[-1])
                slots = slots.reshape(-1)

                self.acl_param_decoder = json.dumps({
                    "seqLen": self.in_ra_seqlens.reshape(-1).tolist(), # seqLen: 序列长度
                    "blockNumsList": self.block_nums_list
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
                self.lm_head_indices_fake
            ]

            if self.speculate_enable:
                self.acl_decoder_operation_inputs.append(torch.tensor(q_lens).to(self.device).to(torch.int32))

            if self.compress_head_enable:
                self.in_reshape_seqlen = torch.ones(input_lengths.shape[0], dtype=torch.int32).npu()
                self.acl_decoder_operation_inputs.append(self.wins_global)
                self.acl_decoder_operation_inputs.append(self.in_ra_seqlens.reshape(-1))
                self.acl_decoder_operation_inputs.append(self.decode_pffset_index.reshape(-1))
                self.acl_decoder_operation_inputs.append(self.razor_offset.reshape(-1, self.razor_offset.shape[-1]))
                self.acl_decoder_operation_inputs.append(self.in_reshape_seqlen)

            return self.acl_decoder_operation_inputs, self.acl_param_decoder
        