#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
from typing import Optional

import torch

from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.log import logger
from atb_llm.models.base.causal_lm import CausalLM
from atb_llm.models.qwen2.modeling_qwen2 import FlashQwenModel
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper


CAUSAL_QWEN2_DEVICE_NAME = "npu"


class Qwen2ForCausalLM(CausalLM):
    def __init__(self, config, weights):
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights)
        self.transformer = FlashQwenModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"] if not config.tie_word_embeddings else ["model.embed_tokens"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )


        self.config = config
        if self.dtype != torch.float16:
            raise ValueError(f"unsupported type: {self.dtype}, 当前仅支持`float16`类型，请修改权重文件config.json中的`torch_dtype`字段")
        self.place_holder = torch.zeros(1, dtype=torch.float16, device=CAUSAL_QWEN2_DEVICE_NAME)
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32, device=CAUSAL_QWEN2_DEVICE_NAME)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=CAUSAL_QWEN2_DEVICE_NAME)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.acl_param = None
        self.acl_operation_inputs = None
        self.ascend_weight = None

    def init_ascend_operations(self, config: Qwen2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_DecoderModel")
        logger.info(">>>> qwen1.5_flash_attention_model is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn', pack_name='c_attn',
            sep_names=None, o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp', pack_name='w2_w1',
            sep_names=None, down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.mlp
                del layer.attn
                del layer.ln_2
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types

        acl_param_dict = {
            "isFA": True,
            "isBF16": False,
            "withEmbedding": True,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "kvQuant": self.config.quantization_config.kv_quant_type is not None
        }
        acl_param_encoder = json.dumps({**acl_param_dict, "isPrefill": True, "supportLcoc": self.lcoc_enable})
        acl_param_decoder = json.dumps({**acl_param_dict, "isPrefill": False, "supportLcoc": False})

        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_kvcache(self, input_ids_or_embedding, past_key_value):
        super().init_kvcache(input_ids_or_embedding, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(
            self,
            input_ids_or_embedding: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[bool],
            max_seq_len: int,
    ):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        cos_table = self.rotary_embedding.get_cos_cached_total()
        sin_table = self.rotary_embedding.get_sin_cached_total()

        if cu_seqlen_prefill:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids_or_embedding.shape[1]] * self.batch_num
            })
            self.acl_operation_inputs = [
                input_ids_or_embedding,  # IN_TENSOR_INPUTIDS
                position_ids,  # IN_TENSOR_POSITIONIDS
                cos_table,  # IN_TENSOR_COSTABLE
                sin_table,  # IN_TENSOR_SINTABLE
                self.mask_full,  # IN_TENSOR_ATTENTIONMASK
                self.place_holder,  # IN_TENSOR_BLOCK_TABLES
                self.place_holder,  # IN_TENSOR_SLOTS
                self.kv_cache_idx,  # IN_TENSOR_KV_CACHE_IDX
                self.token_offset,  # IN_TENSOR_TOKEN_OFFSET
                self.seq_len_encoder,  # IN_TENSOR_SEQ_LENGTHS
                torch.tensor([self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.device),
                # IN_TENSOR_LOGTIS_INDICES
                self.place_holder,  # IN_HOLDER
            ]
        else:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": self.acl_param_seq_len_decoder
            })
            self.acl_operation_inputs = [
                input_ids_or_embedding,  # IN_TENSOR_INPUTIDS
                position_ids,  # IN_TENSOR_POSITIONIDS
                cos_table,  # IN_TENSOR_COSTABLE
                sin_table,  # IN_TENSOR_SINTABLE
                self.mask_full,  # IN_TENSOR_ATTENTIONMASK
                self.place_holder,  # IN_TENSOR_BLOCK_TABLES
                self.place_holder,  # IN_TENSOR_SLOTS
                self.kv_cache_idx,  # IN_TENSOR_KV_CACHE_IDX
                self.token_offset,  # IN_TENSOR_TOKEN_OFFSET
                self.seq_len_decoder,  # IN_TENSOR_SEQ_LENGTHS
                self.lm_head_indices_fake,  # IN_TENSOR_LOGTIS_INDICES
                self.place_holder,  # IN_HOLDER
            ]
        return self.acl_operation_inputs, self.acl_param
