# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
import math
from typing import Optional, List, Tuple

import torch

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from .modeling_qwen2_moe import FlashQwenModel
from .configuration_qwen2_moe import Qwen2MoeConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import AttnWrapper
from ...utils.layers import load_column_multi


class FlashQwen2moeForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        # the Qwen2-moe only support world_size=8
        if config.num_experts_per_tok == 8 and (not weights.process_group.size() == 8):
            msg = f"""
                  unsupported world_size: {weights.process_group.size()}.
                  For Qwen2-57B-A14B-Instruct, Only support world_size: 8. Plase change the world_size value.
                  """
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        super().__init__(config, weights)
        self.model = FlashQwenModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config  # for quantize
        self.attn_mask_fake = self.attn_mask.get_attn_mask(1, dtype=self.dtype, device="npu")
        self.place_holder = torch.tensor([1], dtype=self.dtype, device='npu')

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.tp = True  # default the model is tensor parallel
        if self.tp:
            self.expert_parallel_degree = 1
            self.mask_start_idx = 0
        else:
            self.expert_parallel_degree = self.tp_world_size
            self.mask_start_idx = self.tp_rank
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.final_hidden_states = []
        self.expert_array = None
        self.expert_group = torch.tensor([1], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.acl_param = None
        self.acl_operation_inputs = None
        self.cos_table = None
        self.sin_table = None
        self.acl_param_encoder = None
        self.acl_param_decoder = None
        self.ascend_weight = None

    def init_ascend_operations(self, config: Qwen2MoeConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_MoeDecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_MoeDecoderModel")
        logger.info(">>>> qwen_MoeDecoderModel is called.")

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
            pack_name=['gate_up_proj'],
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=True,
        )
        weight_wrapper = MoeWeightWrapper(
            self.soc_info,
            self.tp_rank,
            attn_wrapper,
            moe_mlp_wrapper,
            self.config.num_experts
        )
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            layer_dict = layer.state_dict()
            # add input layernorm and self_attn weight
            if self.num_experts_per_tok == 4:  # qwen1.5_moe
                weight_wrapper.register_layer_attn(layer, attn_wrapper, self.quantize)
            else:  # qwen2_moe
                weight_wrapper.weights.append(layer.self_attn.pre_norm)
                weight_wrapper.weights.extend([self.place_holder] * 3)
                # load qkv
                weight_wrapper.weights.append(layer.self_attn.qkv_weight)
                weight_wrapper.weights.append(layer.self_attn.qkv_bias)
                weight_wrapper.weights.extend([self.place_holder] * 16)
                # load o_proj
                weight_wrapper.weights.append(layer.self_attn.o_proj)
                weight_wrapper.weights.extend([self.place_holder] * 5)
            # add post norm weights
            weight_wrapper.weights.append(layer_dict["post_attention_layernorm.weight"])
            weight_wrapper.weights.extend([self.place_holder] * 3)
            # add gate weights
            weight_wrapper.weights.append(layer_dict["mlp.gate.weight"])
            weight_wrapper.weights.extend([self.place_holder] * 5)
            # add common experts
            gate_up_list = []
            down_list = []
            for j in range(self.config.num_experts):
                gate_up_list.append(
                    layer_dict[f"mlp.experts.{j}.gate_up_proj.linear.weight"].transpose(0, 1)
                )
                down_list.append(
                    layer_dict[f"mlp.experts.{j}.down_proj.linear.weight"]
                )
                del layer.mlp.experts[j].gate_up_proj.linear.weight
                del layer.mlp.experts[j].down_proj.linear.weight
                torch.npu.empty_cache()
            gate_up_stacked = torch.stack(gate_up_list, dim=0)
            down_stacked = torch.stack(down_list, dim=0) 
            weight_wrapper.weights.append(gate_up_stacked)
            weight_wrapper.weights.extend([self.place_holder] * 5)
            weight_wrapper.weights.append(down_stacked)
            weight_wrapper.weights.extend([self.place_holder] * 5)
            del gate_up_stacked
            del down_stacked
            del gate_up_list
            del down_list
            torch.npu.empty_cache()
            # add shared experts weights
            shared_experts_layer_names = [
                "mlp.shared_expert.gate_up_proj.linear",
                "mlp.shared_expert.down_proj.linear",
            ]
            for layer_name in shared_experts_layer_names:
                weight_wrapper.weights.append(layer_dict[f"{layer_name}.weight"])
                weight_wrapper.weights.extend([self.place_holder] * 5)
            # add shared experts gate weights
            weight_wrapper.weights.append(layer_dict["mlp.shared_expert_gate.weight"])
            weight_wrapper.weights.extend([self.place_holder] * 5)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        ascend_weight_wrapper = self.get_weights()
        self.ascend_weight = ascend_weight_wrapper.weights

        pack_quant_types = [[1, 1] for _ in range(self.config.num_hidden_layers)]
        # attention
        attn_linear_types = [[0, -1, -1, 0, -1, -1] for _ in range(self.config.num_hidden_layers)]
        # shared_expert
        mlp_linear_types = [[0, -1, 0, 0] for _ in range(self.config.num_hidden_layers)]
        # moe
        moe_linear_types = [[0, 0, -1, 0] for _ in range(self.config.num_hidden_layers)]
        # 转置操作
        attn_linear_transpose_types = [[1, -1, -1, 1, -1, -1] for _ in range(self.config.num_hidden_layers)]
        mlp_linear_transpose_types = [[1, -1, 1, -1] for _ in range(self.config.num_hidden_layers)]
        moe_linear_transpose_types = [[1, -1, -1, -1] for _ in range(self.config.num_hidden_layers)]

        acl_param_dict = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "attnLinearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "attnLinearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,  # 是否使用融合算子，对于910B系列已经支持
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_types,
            "expertParallelDegree": self.expert_parallel_degree,
            "rankTableFile": ENV.rank_table_file,
            "numOfExperts": self.num_experts,
            "numOfSelectedExperts": self.config.num_experts_per_tok,
            "routingMethod": 'softmaxTopK',  # 可选专家的方法
        }
        self.acl_param_encoder = json.dumps({**acl_param_dict, "isPrefill": True, "supportLcoc": self.lcoc_enable})
        self.acl_param_decoder = json.dumps({**acl_param_dict, "isPrefill": False, "supportLcoc": False})

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

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
        self.acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        # rope
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype,
            self.device,
            self.max_position_embeddings
        )
        self.cos_table = self.rotary_embedding.get_cos_cached_total()
        self.sin_table = self.rotary_embedding.get_sin_cached_total()
        
        # input_id
        input_length = len(input_ids)
        # expert
        self.expert_array = None
        self.expert_array = torch.tensor(
            [j for j in range(input_length * self.config.num_experts_per_tok)],
            dtype=torch.int32
            ).npu().view(-1)
        
        if is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                attention_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            else:
                attention_mask = self.attn_mask.get_attn_mask(128, kv_cache[0][0].dtype,
                                                              kv_cache[0][0].device)
        else:
            attention_mask = self.attn_mask_fake

        self.acl_operation_inputs = [
            input_ids,  # IN_TENSOR_INPUTIDS
            position_ids,  # IN_TENSOR_POSITIONIDS
            self.cos_table,  # IN_TENSOR_COSEMBED
            self.sin_table,  # IN_TENSOR_SINEMBED
            attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            self.place_holder,  # IN_TENSOR_KV_CACHE_IDX
            self.place_holder,  # IN_TENSOR_TOKEN_OFFSET
            self.place_holder,
            input_lengths.to(torch.int32),  # IN_TENSOR_SEQ_LENGTHS
            lm_head_indices if is_prefill else self.lm_head_indices_fake,  # IN_TENSOR_LOGTIS_INDICES
            self.expert_array,
            self.expert_group,
            self.one_hot,
            self.zero_hot
        ]
        return self.acl_operation_inputs, self.acl_param