# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.


import json
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode

from atb_llm.utils.env import ENV
from ...utils.data.weight_wrapper import AttnWrapper, MlpWrapper, WeightWrapper
from ...utils.layers import load_column_multi
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.layers.norm.fast_layer_norm import NormType
from ..base.flash_causal_lm import FlashForCausalLM
from .config_phi3 import Phi3Config
from .modeling_phi3 import FlashPhi3Model


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with gemma->phi3, Gemma->Phi3
class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = None

    @torch.no_grad()
    def forward(self, seq_len, dtype, device):
        if self.inv_freq is None:
            try:
                self.inv_freq = 1.0 / (
                    self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim)
                )
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e 
        
        position_ids = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Force float32 since bfloat16 loses precision on long contexts
        device_type = device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.outer(position_ids, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class Phi3SuScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling.short_factor
        self.long_factor = config.rope_scaling.long_factor
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, seq_len, dtype, device):
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)

        try:
            inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.double, device=device) / self.dim
            self.inv_freq = (1.0 / (ext_factors * self.base**inv_freq_shape)).to(torch.float)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e 

        position_ids = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Force float32 since bfloat16 loses precision on long contexts
        device_type = device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.outer(position_ids, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            try:
                scale = self.max_position_embeddings / self.original_max_position_embeddings
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e 
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                try:
                    scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
                except ZeroDivisionError as e:
                    raise ZeroDivisionError from e 

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=dtype), sin.to(dtype=dtype)
    

class FlashPhi3ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights, **kwargs)
        self.model = FlashPhi3Model(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.total_head_nums = config.hidden_size // self.head_dim
        self.acl_encoder_operation_inputs = None
        self.acl_decoder_operation_inputs = None

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch(
            "TransdataOperation"
        )
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.cos_embed = None
        self.sin_embed = None
        self.seq_len = 0
        self.acl_param = None
        self.ascend_weight = []

    def init_ascend_operations(self, config: Phi3Config):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(
            "phi3_DecoderModel"
        )
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(
            "phi3_DecoderModel"
        )

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name="input_layernorm",
            wrapper_name="self_attn",
            pack_name="query_key_value",
            sep_names=["q_proj", "k_proj", "v_proj"],
            o_name="o_proj",
        )
        mlp_wrapper = MlpWrapper(
            norm_name="post_attention_layernorm",
            wrapper_name="mlp",
            pack_name="gate_up_proj",
            sep_names=["gate_proj", "up_proj"],
            down_name="down_proj",
        )
        weight_wrapper = WeightWrapper(
            self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper
        )
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        rank_table_file = ENV.rank_table_file
        coder_param = {
            "isUnpadInputs": True,
            "normType": NormType.RMS_NORM,
            "normEps": self.config.rms_norm_eps,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableAddNorm": False,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or rank_table_file else "lccl",
            "rankTableFile": rank_table_file,
        }
        encoder_param = {
            **coder_param,
            "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
        }
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

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
        self.seq_len = torch.max(position_ids) + 1
        atten_mask = None

        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(
                self.max_base_len, self.dtype, self.device
            )
            self.init_cos_sin_table(
                self.max_position_embeddings, self.head_dim, self.dtype, self.device
            )

            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(
                    range(input_ids.shape[0]),
                    dtype=torch.int64,
                    device=input_ids.device,
                )
            self.acl_param = json.dumps({"seqLen": input_lengths.tolist()})
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
            self.acl_param = json.dumps({"seqLen": input_lengths.tolist()})
            atten_mask = self.attn_mask_fake
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

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        if self.rope_given_inv_feq_str is None and self.rope_vanilla_theta is None:
            self._init_rope_cos_sin(dim, dtype, device)
        else:
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len,
                dim,
                dtype,
                device
            )

    # 固定基频: rope_theta
    # 自定义基频: rope_given_inv_feq_str
    # 分段基频: rope_theta/rope_given_inv_feq_str + rope_vanilla_theta + rope_keep_local_base_windows
    def _get_cos_sin_table(
        self,
        max_seq_len,
        dim,
        dtype,
        device,
    ):
        if self.rope_given_inv_feq_str:
            inv_freq = torch.FloatTensor(
                [float(invf) for invf in self.rope_given_inv_feq_str.split(",")], device=device
            )
            if len(inv_freq) != dim // 2:
                raise AssertionError("given_inv_feq_str: length not match head_dim/2")
        else:
            try:
                inv_freq = 1.0 / (
                    self.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
                )
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e

        seq = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(seq, inv_freq)

        if self.rope_keep_local_base_windows:
            self.rope_keep_local_base_windows = [int(w) for w in self.rope_keep_local_base_windows.split(",")]
            if len(self.rope_keep_local_base_windows) != dim // 2:
                raise AssertionError(
                    "keep_local_base_windows: length not match head_dim/2"
                )
            try:
                inv_freq_base = 1.0 / (
                    self.rope_vanilla_theta
                    ** (torch.arange(0, dim, 2, device=device).float() / dim)
                )
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e       
            freqs_base = torch.outer(seq, inv_freq_base)
            freqs_after_window = freqs + torch.tensor(self.rope_keep_local_base_windows) * (
                inv_freq_base - inv_freq
            )
            for idx, i_keep_local_base_window in enumerate(self.rope_keep_local_base_windows):
                freqs[:, idx] = torch.cat(
                    (
                        freqs_base[:i_keep_local_base_window, idx],
                        freqs_after_window[i_keep_local_base_window:, idx],
                    )
                )

        # Different from paper, but it uses a different permutation in order to obtain the same calculation（ks）
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.rope_mscale).to(dtype).to(device), (emb.sin() * self.rope_mscale).to(
            dtype
        ).to(device)

    def _init_rope_cos_sin(self, dim, dtype, device):
        if self.config.rope_scaling:
            scaling_type = self.config.rope_scaling.type
            if scaling_type == "su":
                rotary_embed = Phi3SuScaledRotaryEmbedding(
                    dim,
                    self.config
                )
                self.cos_embed, self.sin_embed = rotary_embed.forward(
                    self.config.original_max_position_embeddings,
                    dtype,
                    device
                )
            else:
                msg = f"Unknown RoPE scaling type {scaling_type}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
        else:
            msg = "Check in config.json: rope_scaling must not be None"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        