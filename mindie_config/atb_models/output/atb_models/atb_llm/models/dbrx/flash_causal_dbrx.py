# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
import math
import os
from typing import Optional, List, Tuple
from atb_llm.utils.env import ENV
import torch

from atb_llm.models.dbrx.modeling_dbrx import FlashDbrxModel
from atb_llm.models.dbrx.configuration_dbrx import DbrxConfig
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.layers.norm.fast_layer_norm import NormType


class FlashDbrxForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        # shared_expert down is not support transpose
        os.environ["ATB_LLM_ENABLE_AUTO_TRANSPOSE"] = "0"
        super().__init__(config, weights)
        self.model = FlashDbrxModel(config, weights, **kwargs)

        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config  # for quantize
        if self.dtype != torch.float16:
            raise ValueError(
                f"unsupported type: {self.dtype}, 当前仅支持`float16`类型，请修改权重文件config.json中的`torch_dtype`字段")
        self.attn_mask_fake = self.attn_mask.get_attn_mask(1, dtype=torch.float16, device="npu")
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.expert_parallel_degree = 1
        self.mask_start_idx = 0

        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k
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
        self.max_position_embeddings = config.max_seq_len
        self.rope_theta = 500000.0

    def init_ascend_operations(self, config: DbrxConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("dbrx_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("dbrx_DecoderModel")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=None,
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name=None,
            sep_names=["w1", "v1", "w2"],
            down_name='w2'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)

        weight_wrapper.register_embedding(self.model.wte)

        for i in range(self.num_layers):
            layer = self.model.layers[i]
            layer_dict = layer.state_dict()
            weight_wrapper.register_layer_attn(layer, attn_wrapper, self.quantize)

            # add post norm weights
            weight_wrapper.weights.append(layer_dict["ln_2.weight"])
            weight_wrapper.weights.append(layer_dict["ln_2.bias"])
            weight_wrapper.weights.extend([self.place_holder] * 2)

            # add gate weights
            weight_wrapper.weights.append(layer_dict["mlp.gate.weight"])
            weight_wrapper.weights.extend([self.place_holder] * 5)

            w1_v1_list = []
            for expert_id in range(self.num_experts):
                key = "mlp.w1_v1." + str(expert_id) + ".weight"
                current_weight = layer_dict[key]
                w1_v1_list.append(current_weight)
            weight_wrapper.weights.append(torch.stack([w1_v1 for w1_v1 in w1_v1_list], dim=0))

            weight_wrapper.weights.extend([self.place_holder] * 5)

            w2_list = []
            for expert_id in range(self.num_experts):
                key = "mlp.w2." + str(expert_id) + ".weight"
                current_weight = layer_dict[key]
                w2_list.append(current_weight)

            weight_wrapper.weights.append(torch.stack([w2 for w2 in w2_list], dim=0))

            weight_wrapper.weights.extend([self.place_holder] * 5)
            del layer.mlp

            torch.npu.empty_cache()

            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp

        weight_wrapper.register_model_norm(self.model.ln_f)

        weight_wrapper.register_model_lmhead(self.lm_head)

        return weight_wrapper

    def init_ascend_weight(self):
        ascend_weight_wrapper = self.get_weights()
        self.ascend_weight = ascend_weight_wrapper.weights

        pack_quant_types_list = [[1, 1, 1, 1] for _ in range(self.config.num_hidden_layers)]
        attn_linear_types_list = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.config.num_hidden_layers)]
        mlp_linear_types_list = [[0, 0, 0, 0] for _ in range(self.config.num_hidden_layers)]
        moe_linear_types_list = [[0, 0, 0, 0] for _ in range(self.config.num_hidden_layers)]

        attn_linear_transpose_types_list = [[1, -1, -1, 1, -1, -1, -1] for _ in range(self.config.num_hidden_layers)]
        mlp_linear_transpose_types_list = [[1, 1, 1, 1] for _ in range(self.config.num_hidden_layers)]
        moe_linear_transpose_types_list = [[1, 1, 1, 1] for _ in range(self.config.num_hidden_layers)]

        acl_param_dict = {
            "normEps": 1e-5,
            "normType": NormType.LAYER_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": False,
            "packQuantType": pack_quant_types_list,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearQuantType": attn_linear_types_list,
            "mlpLinearQuantType": mlp_linear_types_list,
            "moeLinearQuantType": moe_linear_types_list,
            "linearTransposeType": attn_linear_transpose_types_list,
            "mlpLinearTransposeType": mlp_linear_transpose_types_list,
            "moeLinearTransposeType": moe_linear_transpose_types_list,
            "lmHeadTransposeType": 1,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "expertParallelDegree": self.expert_parallel_degree,
            "numOfExperts": self.config.ffn_config.moe_num_experts,
            "routingMethod": 'softMaxTopK' if self.soc_info.need_nz else 'integratedSoftmaxTopK',
            "numOfSelectedExperts": self.config.ffn_config.moe_top_k,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "enableAddNorm": False,
            "normHasBias": True,
            "enableFusedRouting": False
        }
        encoder_param = {
            **acl_param_dict,
            "isPrefill": True,
            "enableLcoc": False,
            "skipWordEmbedding": False
        }
        decoder_param = {
            **acl_param_dict,
            "isPrefill": False,
            "enableLcoc": False,
            "skipWordEmbedding": False
        }

        self.acl_param_encoder = json.dumps({**encoder_param})
        self.acl_param_decoder = json.dumps({**decoder_param})

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

        input_length = input_ids.shape[0]

        self.expert_array = torch.arange(input_length * self.num_experts_per_tok, dtype=torch.int32).npu().view(-1)

        if is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.get_cos_sin()
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

    def get_cos_sin(self):
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype,
            self.device,
            self.max_position_embeddings
        )
        self.cos_table = self.rotary_embedding.get_cos_cached_total()
        self.sin_table = self.rotary_embedding.get_sin_cached_total()


if __name__ == "__main__":
    test_config = DbrxConfig()
    test_weights = None
    model = FlashDbrxForCausalLM(test_config, test_weights)
