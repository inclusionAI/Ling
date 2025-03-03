# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional, List, Tuple

import torch

from atb_llm.utils.env import ENV
from .modeling_aquila import FlashAquilaModel
from .config_aquila import AquilaConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.layers import load_column_multi
from ...utils.dist import get_rank_table_file
from ...utils.layers.norm.fast_layer_norm import NormType


class FlashAquilaForCausalLM(FlashForCausalLM):
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

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

    def init_ascend_operations(self, config: AquilaConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("aquila_7b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("aquila_7b_DecoderModel")

    def init_ascend_weight(self):
        weight_wrapper = super().get_weight_wrapper()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
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
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or rank_table_file else "lccl",
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
        self.acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        self.acl_operation_inputs = [
            input_ids,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32)
        ]
        atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype, kv_cache[0][0].device)
        if self.soc_info.need_nz:
            atten_mask = self.transdata_operation.execute([atten_mask])[0]

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        self.acl_operation_inputs.insert(4, atten_mask)
        self.acl_operation_inputs.append(lm_head_indices.to(torch.int64) if is_prefill else self.lm_head_indices_fake)

        return self.acl_operation_inputs, self.acl_param