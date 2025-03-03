# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional, List, Union, Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base.causal_lm import CausalLM
from .modeling_aquila import FlashAquilaModel
from .config_aquila import AquilaConfig
from ...utils.layers import load_column_multi
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.dist import get_rank_table_file


class AquilaForCausalLM(CausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.model = FlashAquilaModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.in_tensor_length = 12

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

    def init_ascend_operations(self, config: AquilaConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("aquila_7b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("aquila_7b_DecoderModel")

    def init_ascend_weight(self):
        weight_wrapper = super().get_weight_wrapper()
        linear_types = weight_wrapper.linear_type
        self.ascend_weight = weight_wrapper.weights
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        rank_table_file = get_rank_table_file()
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "isUnpadInputs": True,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": True,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": not self.soc_info.need_nz,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "rankTableFile": rank_table_file,
        }
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
        encoder_param = {**coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_kvcache(self, input_ids_or_embedding, past_key_value):
        super().init_kvcache(input_ids_or_embedding, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[bool],
                                  max_seq_len: int,
                                  ):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                         self.device,
                                                         max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        self.acl_param = json.dumps({
            "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
            "seqLen": [input_ids.shape[1]] * self.batch_num if cu_seqlen_prefill \
                else self.acl_param_seq_len_decoder
        })

        self.acl_operation_inputs = [
            input_ids,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            self.mask_full,
            self.placeholder,
            self.placeholder,
            self.kv_cache_idx,
            self.token_offset,
            self.placeholder
        ]

        self.acl_operation_inputs.extend([
            self.seq_len_encoder if cu_seqlen_prefill else self.seq_len_decoder,
            torch.tensor([self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.device) \
                if cu_seqlen_prefill else self.lm_head_indices_fake
        ])

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                               labels, use_cache, output_attentions, output_hidden_states, return_dict)

