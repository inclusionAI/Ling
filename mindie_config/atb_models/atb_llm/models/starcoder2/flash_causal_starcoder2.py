# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
from typing import Optional, List, Tuple

import torch
import torch.distributed
from atb_llm.utils.layers import TensorParallelHead

from .config_starcoder2 import Starcoder2Config
from .modeling_starcoder2 import FlashStarcoder2Model
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType


CPP_STARCODER2_MODEL_CLASS_NAME = "starcoder2_DecoderModel"


class FlashStarcoder2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.model = FlashStarcoder2Model(config, weights)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads // self.tp_world_size  # 48
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.layer_norm_epsilon = config.norm_epsilon
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.acl_param = None
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        self.lm_head = TensorParallelHead.load(
            config,
            prefix="model.embed_tokens" if config.tie_word_embeddings else "lm_head",
            weights=weights,
            is_norm=True
        )
        self.placeholder = torch.ones(1, dtype=self.dtype, device="npu")
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                         self.device,
                                                         self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: Starcoder2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_STARCODER2_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_STARCODER2_MODEL_CLASS_NAME)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='c_fc',
            sep_names=None,
            down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
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
        return (weight_wrapper)

    def init_ascend_weight(self):
        weight = self.get_weights()
        self.ascend_weight, linear_types, pack_quant_configs, linear_transpose_types =\
            weight.weights, weight.linear_type, weight.pack_quant_type, weight.linear_transpose_types
        coder_param = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "isUnpadInputs": True,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "normEps": self.layer_norm_epsilon,
            "normType": NormType.LAYER_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.num_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "packQuantType": pack_quant_configs,
            "linearHasBias": [[True, True, True, True]] * self.num_layers,
            "linearQuantType": linear_types,
            # PreLayerNorm暂不支持量化以及bf16
            "enableAddNorm": self.quantize is None and self.dtype == torch.float16
        }
        encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": self.lcoc_enable}
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

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
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                         self.device,
                                                         self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                      kv_cache[0][0].device)

            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
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
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                self.lm_head_indices_fake,
            ]
            return self.acl_decoder_operation_inputs, self.acl_param
