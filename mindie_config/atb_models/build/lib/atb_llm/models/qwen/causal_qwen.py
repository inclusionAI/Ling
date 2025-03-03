#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
import math
import importlib
from typing import Optional

import torch

from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV
from atb_llm.models.base.causal_lm import CausalLM
from atb_llm.models.qwen.modeling_qwen import FlashQwenModel
from atb_llm.models.qwen.config_qwen import QwenConfig
from atb_llm.models.qwen.base_qwen import QwenRotaryEmbedding
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper


QWEN_DEVICE_NAME = "npu"


class QwenForCausalLM(CausalLM):
    def __init__(self, config, weights):
        super().__init__(config, weights)
        self.transformer = FlashQwenModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["model.embed_tokens"] if config.tie_word_embeddings else ["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config
        if self.dtype != torch.float16:
            msg = f"unsupported type: {self.dtype}, just support `float16`, please modify the `torch_dtype`"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        self.place_holder = torch.zeros(1, dtype=torch.float16, device=QWEN_DEVICE_NAME)
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32, device=QWEN_DEVICE_NAME)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=QWEN_DEVICE_NAME)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.acl_param = None
        self.acl_operation_inputs = None
        self.ascend_weight = None

        self.long_seq_enable = ENV.long_seq_enable
        if self.long_seq_enable:
            logn_list = [
                math.log(i, config.seq_length) if i > config.seq_length else 1
                for i in range(1, 32768)
            ]
            self.logn_tensor = torch.tensor(logn_list,
                                            dtype=torch.float16, device=QWEN_DEVICE_NAME)[None, :, None, None]
            if config.rotary_pct == 1.0:
                self.rotary_ndims = None
            elif config.rotary_pct < 1:
                self.rotary_ndims = int(config.kv_channels * config.rotary_pct)
            else:
                logger.error("rotary_pct > 1", ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise RuntimeError("rotary_pct > 1")(msg)
            self.rotary_emb = QwenRotaryEmbedding(
                self.rotary_ndims if self.rotary_ndims else config.kv_channels,
                base=config.rotary_emb_base
            )

    def init_ascend_operations(self, config: QwenConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_DecoderModel")
        logger.info(">>>> qwen_flash_attention_model is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(norm_name='ln_1',
            wrapper_name='attn', pack_name='c_attn',
            sep_names=None, o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(norm_name='ln_2',
            wrapper_name='mlp', pack_name='w2_w1',
            sep_names=None, down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.attn
                del layer.mlp
                del layer.ln_2

        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_kvcache(self, input_ids_or_embedding, past_key_value):
        super().init_kvcache(input_ids_or_embedding, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        pack_quant_configs = weight_wrapper.pack_quant_type
        
        acl_param_dict = {
            "isFA": True, "isBF16": False,
            "withEmbedding": True, "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rmsNormEps": self.config.layer_norm_epsilon,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank, "worldSize": self.tp_world_size,
            "enableLogN": True if self.long_seq_enable else False,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "quantGroupSize": self.config.quantization_config.group_size,
            "kvQuant": False
        }
        acl_param_encoder = json.dumps({**acl_param_dict, "isPrefill": True, "supportLcoc": self.lcoc_enable})
        acl_param_decoder = json.dumps({**acl_param_dict, "isPrefill": False, "supportLcoc": False})
        
        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)
        
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)
    
    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / self.config.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def prepare_inputs_for_ascend(
        self,
        input_ids_or_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[bool],
        max_seq_len: int,
    ):
        cos_embed, sin_embed = None, None
        cos_table, sin_table = None, None
        logn_tensor = None
        if self.long_seq_enable:
            seq_start = self.token_offset[0] - input_ids_or_embedding.shape[-1]
            seq_end = self.token_offset[0]
            logn_tensor = self.logn_tensor[:, seq_start: seq_end, :, :]
            logn_tensor = logn_tensor.repeat(input_ids_or_embedding.shape[0], 1,
                                             self.num_attention_heads, self.head_size)

            if not self.config.use_dynamic_ntk:
                ntk_alpha_list = [1.0]
            elif self.token_offset[0] != input_ids_or_embedding.shape[1]:
                ntk_alpha_list = self.rotary_emb.get_ntk_alpha_list()
            else:
                ntk_alpha_list = []
                if self.token_offset[0] > self.config.seq_length:
                    true_seq_lens = torch.tensor(
                                                [input_ids_or_embedding[i].shape[0] 
                                                for i in range(input_ids_or_embedding.shape[0])]
                                                )
                    for i in range(input_ids_or_embedding.shape[0]):
                        true_seq_len = true_seq_lens[i].item()
                        ntk_alpha = self.get_ntk_alpha(true_seq_len)
                        ntk_alpha_list.append(ntk_alpha)
                else:
                    ntk_alpha = self.get_ntk_alpha(self.token_offset[0])
                    ntk_alpha_list.append(ntk_alpha)
            self.rotary_emb.set_ntk_alpha_list(ntk_alpha_list) 
            rotary_pos_emb_list = [
                self.rotary_emb(self.token_offset[0], ntk_alpha=ntk_alpha)
                for ntk_alpha in ntk_alpha_list
            ]
            rotary_pos_emb = rotary_pos_emb_list[0]
            cos_embed = rotary_pos_emb[0][:, -input_ids_or_embedding.shape[1]:, :, :]
            sin_embed = rotary_pos_emb[1][:, -input_ids_or_embedding.shape[1]:, :, :]
            cos_embed = cos_embed.repeat(input_ids_or_embedding.shape[0], 1, 1, 1)[0].squeeze(1)
            sin_embed = sin_embed.repeat(input_ids_or_embedding.shape[0], 1, 1, 1)[0].squeeze(1)
        else:
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
            cos_table = self.rotary_embedding.get_cos_cached_total()
            sin_table = self.rotary_embedding.get_sin_cached_total()

        if cu_seqlen_prefill:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids_or_embedding.shape[1]] * self.batch_num
            })
            self.acl_operation_inputs = [
                input_ids_or_embedding,
                position_ids,
                cos_embed if self.long_seq_enable else cos_table,
                sin_embed if self.long_seq_enable else sin_table,
                self.mask_full,
                self.place_holder,
                logn_tensor if self.long_seq_enable else self.place_holder,
                self.kv_cache_idx,
                self.token_offset,
                self.seq_len_encoder,
                torch.tensor([self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.device),
                self.place_holder,
            ]
        else:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": self.acl_param_seq_len_decoder
            })
            self.acl_operation_inputs = [
                input_ids_or_embedding,
                position_ids,
                cos_embed if self.long_seq_enable else cos_table,
                sin_embed if self.long_seq_enable else sin_table,
                self.mask_full,
                self.place_holder,
                logn_tensor if self.long_seq_enable else self.place_holder,
                self.kv_cache_idx,
                self.token_offset,
                self.seq_len_decoder,
                self.lm_head_indices_fake,
                self.place_holder,
            ]
        return self.acl_operation_inputs, self.acl_param
