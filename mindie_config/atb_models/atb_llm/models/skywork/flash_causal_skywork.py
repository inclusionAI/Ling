# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional, List, Tuple

import torch

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.skywork.config_skywork import SkyworkConfig
from atb_llm.models.skywork.modeling_skywork import FlashSkyworkModel
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class FlashSkyworkForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        # called the:
        # self.init_ascend_operations
        super().__init__(config, weights, **kwargs)
        self.model = FlashSkyworkModel(config, weights)
        self.config = config
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config
        self.in_tensor_length = 16
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        
        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.prefill_skip_word_embedding = False
        self.decoder_skip_word_embedding = False
        self.padding_idx = config.pad_token_id
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.hidden_dim = config.hidden_size
        self.expert_array = []
        self.expert_group = torch.tensor([1], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.tp = config.tp if config.tp else True # False
        if self.tp:
            self.expert_parallel_degree = 1
        else:
            self.expert_parallel_degree = self.tp_world_size
        self.ascend_weight_wrapper = None
        self.ascend_weight = None

    # called by super().prepare_inputs_for_ascend
    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: SkyworkConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("skywork_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("skywork_DecoderModel")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='block_sparse_moe',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=False
        )
        weight_wrapper = MoeWeightWrapper(self.soc_info, self.tp_rank,
                                          attn_wrapper, moe_mlp_wrapper,
                                          self.config.n_routed_experts)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            if self.tp:
                weight_wrapper.register_moe_layer(layer, self.quantize)
                del layer.block_sparse_moe
                torch.npu.empty_cache()
            else:
                msg = "ERROR: Skywork does not support expert parallel!"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                torch.npu.empty_cache()
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        ascend_weight_wrapper = self.get_weights()
        self.ascend_weight = ascend_weight_wrapper.weights

        pack_quant_types = ascend_weight_wrapper.pack_quant_type
        attn_linear_types = ascend_weight_wrapper.attn_linear_types
        mlp_linear_types = ascend_weight_wrapper.mlp_linear_types
        moe_linear_types = ascend_weight_wrapper.moe_linear_types
        attn_linear_transpose_types = ascend_weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = ascend_weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = ascend_weight_wrapper.moe_linear_transpose_types
        for i in range(self.num_layers):
            attn_linear_types[i].append(attn_linear_types[i][-1])
            attn_linear_transpose_types[i].append(-1)
        
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": False,
            "packQuantType": pack_quant_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "linearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "expertParallelDegree": self.expert_parallel_degree,
            "numOfExperts": self.config.n_routed_experts,
            "routingMethod": 'softMaxTopK' if self.soc_info.need_nz else 'integratedSoftmaxTopK',
            "numOfSelectedExperts": self.config.num_experts_per_tok,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "enableAddNorm": False,
            "normHasBias": False,
            "enableFusedRouting": False
        }
        if coder_param["routingMethod"] not in ['softMaxTopK', 'integratedSoftmaxTopK']:
            msg = "The routingMethod chosen is not valid, please choose among the following:\n \
                  'softMaxTopK': regular routing method with softmax and topk-sort operators\n \
                  'integratedSoftmaxTopK': routing method with the integration of softmax and topk-sort operators\n \
                  'deviceLimited': device-limited routing method (e.g. deepseekv2); \
                  invalid for skywork MoE and Deepseekv1"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        encoder_param = {
            **coder_param, "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "skipWordEmbedding": self.prefill_skip_word_embedding
        }
        decoder_param = {
            **coder_param, "isPrefill": False,
            "enableLcoc": False,
            "skipWordEmbedding": self.decoder_skip_word_embedding
        }
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    # called by super().forward()
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

        input_length = len(input_ids)
        self.expert_array = None
        self.expert_array = torch.tensor(
            [j for j in range(input_length * self.config.num_experts_per_tok)],
            dtype=torch.int32).npu().view(-1)

        if is_prefill:
            if self.soc_info.need_nz:
                atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                          kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                torch.where(atten_mask == -torch.inf, 1, atten_mask) if self.dtype == torch.bfloat16 else atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64),
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
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
                torch.zeros(input_lengths.size(0),
                            self.num_attention_heads,
                            1, input_lengths.max().item(),
                            dtype=self.dtype,
                            device=self.device) if self.dtype == torch.bfloat16 else self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                self.lm_head_indices_fake,
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
            ]

            return self.acl_decoder_operation_inputs, self.acl_param


