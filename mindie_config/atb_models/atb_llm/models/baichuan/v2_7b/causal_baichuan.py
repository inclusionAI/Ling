#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
from typing import Optional, List, Union, Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from atb_llm.utils.layers import load_column_multi
from atb_llm.models.baichuan.v2_7b.config_baichuan import BaichuanConfig
from atb_llm.models.base.causal_lm import CausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..modeling_baichuan import FlashBaichuanModel


class BaichuanForCausalLM(CausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
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
        self.placeholder = torch.zeros(1, dtype=self.dtype).npu()
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

        self.in_tensor_length = 12
        self.acl_encoder_operation_inputs: list = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list = [None] * self.in_tensor_length

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.sin_embed = None
        self.cos_embed = None

    def init_ascend_operations(self, config: BaichuanConfig):
        # 初始化模型
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
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isUnpadInputs": False,
            "isFA": True,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": not self.soc_info.need_nz,
            "enableKvQuant": False,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "enableAddNorm": False,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "rankTableFile": ENV.rank_table_file
        }
        encoder_param = {**coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable}
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_kvcache(self, input_ids_or_embedding, past_key_value):
        super().init_kvcache(input_ids_or_embedding, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(self,
                                  input_ids_or_embedding: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[bool],
                                  max_seq_len: int,
                                  ):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                         self.device,
                                                         max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        if cu_seqlen_prefill:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids_or_embedding.shape[1]] * self.batch_num
            })
            self.acl_encoder_operation_inputs[0] = input_ids_or_embedding
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = self.cos_embed
            self.acl_encoder_operation_inputs[3] = self.sin_embed
            self.acl_encoder_operation_inputs[4] = self.mask_full
            self.acl_encoder_operation_inputs[5] = self.placeholder
            self.acl_encoder_operation_inputs[6] = self.placeholder
            self.acl_encoder_operation_inputs[7] = self.kv_cache_idx
            self.acl_encoder_operation_inputs[8] = self.token_offset
            self.acl_encoder_operation_inputs[9] = self.placeholder
            self.acl_encoder_operation_inputs[10] = self.seq_len_encoder
            self.acl_encoder_operation_inputs[11] = torch.tensor(
                [self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.device)
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": self.acl_param_seq_len_decoder
            })
            self.acl_decoder_operation_inputs[0] = input_ids_or_embedding
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.cos_embed
            self.acl_decoder_operation_inputs[3] = self.sin_embed
            self.acl_decoder_operation_inputs[4] = self.mask_full
            self.acl_decoder_operation_inputs[5] = self.placeholder
            self.acl_decoder_operation_inputs[6] = self.placeholder
            self.acl_decoder_operation_inputs[7] = self.kv_cache_idx
            self.acl_decoder_operation_inputs[8] = self.token_offset
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = self.seq_len_decoder
            self.acl_decoder_operation_inputs[11] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param

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
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                               labels, use_cache, output_attentions, output_hidden_states, return_dict)
