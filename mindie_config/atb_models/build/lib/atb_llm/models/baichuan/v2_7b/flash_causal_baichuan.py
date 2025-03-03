# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional, List, Tuple

import torch
import torch.distributed
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import (
    load_column_multi,
    PositionRotaryEmbedding
)
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.env import ENV
from ..modeling_baichuan import FlashBaichuanModel


class FlashBaichuanForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.base = 1e6 if self.num_layers == 60 else 10000
        self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.base,
                                                               device="cpu", scaling_factor=self.scaling_factor) \
            .to(self.device)
        self.model = FlashBaichuanModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
            norm=config.vocab_size == 125696
        )
        self.config = config
        if self.dtype != torch.float16:
            error_msg = f"Unsupported type: {self.dtype}. " \
                        f"Only the `float16` type is supported. Modify the `torch_dtype` field in the config.json."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)
        self.place_holder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")
        self.attn_backend = OpBackend.ATB if self.config.quantization_config.kv_quant_type is None else OpBackend.ATB

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)
        self.acl_operation_inputs = None

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_7b_PagedAttentionQuantModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_7b_PagedAttentionQuantModel")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='W_pack',
            sep_names=None,
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
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
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return (weight_wrapper.weights, weight_wrapper.linear_type,
                weight_wrapper.pack_quant_type, weight_wrapper.linear_transpose_types)

    def init_ascend_weight(self):
        self.ascend_weight, linear_types, pack_quant_configs, linear_transpose_types = self.get_weights()
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": not self.soc_info.need_nz,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "enableAddNorm": False,
            "attnBackend": self.attn_backend,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file
        }
        encoder_param = {**coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable}
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
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
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype,
                                                      kv_cache[0][0].device)
            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        else:
            if self.attn_backend == OpBackend.ACLNN:
                atten_mask = self.attn_mask.get_decode_attn_mask(input_lengths,
                                                                 self.max_position_embeddings, torch.bool, self.device)
            else:
                atten_mask = self.attn_mask_fake

        self.acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
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
            lm_head_indices if is_prefill else self.lm_head_indices_fake,
        ]

        for ind, item in enumerate(self.acl_operation_inputs):
            logger.debug(f"{ind} {item.device=}")
        return self.acl_operation_inputs, self.acl_param
