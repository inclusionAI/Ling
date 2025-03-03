# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
from typing import Optional, List, Tuple

import torch
import torch.distributed
from atb_llm.utils.layers import TensorParallelHead

from .config_starcoder import StarcoderConfig
from .modeling_starcoder import FlashStarcoderModel
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType


CPP_STARCODER_MODEL_CLASS_NAME = "starcoder_DecoderModel"


class FlashStarcoderForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads // self.tp_world_size  # 48
        self.max_seq_len_every_batch = config.max_position_embeddings
        self.num_key_value_heads = config.num_key_value_heads
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.acl_param = None
        self.model = FlashStarcoderModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config,
            prefix="lm_head",
            weights=weights,
            is_norm=True
        )
        self.placeholder = torch.ones(1, dtype=self.dtype, device="npu")

    def init_ascend_operations(self, config: StarcoderConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_STARCODER_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_STARCODER_MODEL_CLASS_NAME)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='self_attn',
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
        weight_wrapper.register_embedding(self.model.wte)
        weight_wrapper.register_embedding(self.model.wpe)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
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
            "isUnpadInputs": True,
            "isEmbeddingParallel": False,
            "positionEmbeddingType": PositionEmbeddingType.ABSOLUTE,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False,
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
        }
        encoder_param = {**coder_param, "isPrefill": True}
        decoder_param = {**coder_param, "isPrefill": False}
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
                self.placeholder,
                self.placeholder,
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
                self.placeholder,
                self.placeholder,
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
