import json
import math
from typing import Optional, List, Tuple

import torch
import torch_npu
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.deepseek.config_deepseek import DeepseekConfig
from atb_llm.models.deepseek.modeling_deepseek import FlashDeepseekModel
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode

from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)
from atb_llm.utils.layers.norm.fast_layer_norm import NormType

_COMMON_EXPERTS_NUM = 64
_DECODER_MODEL = "deepseek_DecoderModel"
_SUPPORT_LCOC = "enableLcoc"
_SUPPORT_SPECULATE = "enableSpeculate"
_IS_PREFILL = "isPrefill"


class FlashDeepseekForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.acl_decoder_regression_operation = None
        super().__init__(config, weights, **kwargs)
        self.model = FlashDeepseekModel(config, weights)
        self.config = config
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
            norm_head=True,
        )
        self.config = config
        self.in_tensor_length = 16
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.padding_idx = config.pad_token_id
        self.embed_tokens = TensorEmbedding(
            prefix="model.word_embeddings", weights=weights
        )
        self.hidden_dim = config.hidden_size
        self.expert_array = []
        self.expert_group = torch.tensor([0], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.final_bias = torch.zeros([self.config.num_experts, self.config.hidden_size], dtype=self.dtype).npu()
        self.num_of_experts = config.num_experts
        self.num_of_selected_experts = [config.num_experts_per_tok]
        self.tp = config.tp if config.tp else True # Defaulting the model to tensor parallel
        self.first_k_dense_replace = config.first_k_dense_replace if config.first_k_dense_replace else 0
        self.n_shared_experts = config.num_shared_experts if config.num_shared_experts else 0
        self.norm_topk_prob = config.norm_topk_prob if config.norm_topk_prob else False
        if self.tp:
            self.expert_parallel_degree = 1
        else:
            self.expert_parallel_degree = self.tp_world_size
        self.ascend_weight = None

        if self.prefix_cache_enable:
            self.acl_decoder_regression_operation_inputs = []
        self.enable_fused_routing = False if self.soc_info.need_nz else True
        if config.model_type == "bailing_moe" and config.num_hidden_layers == 88:
            self.num_attention_heads = 7

    # called by super().prepare_inputs_for_ascend
    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: DeepseekConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(_DECODER_MODEL)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(_DECODER_MODEL)
        if self.prefix_cache_enable:
            self.acl_decoder_regression_operation = torch.classes.ModelTorch.ModelTorch(_DECODER_MODEL)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='dense'
        )
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=(self.n_shared_experts > 0)
        )
        weight_wrapper = MoeWeightWrapper(self.soc_info, self.tp_rank,
                                          attn_wrapper, moe_mlp_wrapper,
                                          self.num_of_experts)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            if i < self.first_k_dense_replace:
                weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=True)
            else:
                if self.tp:
                    weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=False)
                    del layer.mlp
                    torch.npu.empty_cache()

                else:
                    msg = "ERROR: DeepSeek does not support expert parallel!"
                    logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                    raise ValueError(msg)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                torch.npu.empty_cache()
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        pack_quant_configs = weight_wrapper.pack_quant_type

        attn_linear_types = weight_wrapper.attn_linear_types
        mlp_linear_types = weight_wrapper.mlp_linear_types
        moe_linear_types = weight_wrapper.moe_linear_types

        attn_linear_transpose_types = weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = weight_wrapper.moe_linear_transpose_types

        # compatible with linearQuantType
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
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "linearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else False,
            'hasSharedExpert': True if self.n_shared_experts > 0 else False,
            'hasSharedExpertGate': False,
            "rank": self.tp_rank,
            "expertParallelDegree": self.expert_parallel_degree,
            "numOfExperts": self.num_of_experts,
            "numOfGroups": 8,
            "routingMethod": 'softMaxTopK' if self.soc_info.need_nz else 'integratedSoftmaxTopK',
            "processLogits": 'normalization' if self.norm_topk_prob else 'none',
            "firstKDenseReplace": self.first_k_dense_replace,
            "numOfSharedExperts": self.n_shared_experts,
            "numOfSelectedExperts": self.num_of_selected_experts,
            "numOfSelectedGroups": 3,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "enableAddNorm": False,
            "normHasBias": False,
            "enableFusedRouting": self.enable_fused_routing
        }
        if coder_param["routingMethod"] not in ['softMaxTopK', 'integratedSoftmaxTopK']:
            msg = "The routingMethod chosen is not valid, please choose among the following:\n \
                  'softMaxTopK': regular routing method with softmax and topk-sort operators\n \
                  'integratedSoftmaxTopK': routing method with the integration of softmax and topk-sort operators\n \
                  'deviceLimited': device-limited routing method (e.g. deepseekv2); \
                  invalid for Mixtral MoE and Deepseekv1"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        encoder_param = {
            **coder_param, _IS_PREFILL: True, _SUPPORT_LCOC: self.lcoc_enable,
            _SUPPORT_SPECULATE: False, "enableSplitFuse": self.split_fuse_enable
        }
        decoder_param = {
            **coder_param, _IS_PREFILL: False, _SUPPORT_LCOC: False,
            _SUPPORT_SPECULATE: self.speculate_enable, "enablePrefixCache": self.prefix_cache_enable
        }
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

        if self.prefix_cache_enable:
            decoder_regression_param = {
                **coder_param, _IS_PREFILL: False, _SUPPORT_LCOC: False,
                _SUPPORT_SPECULATE: False
            }
            self.acl_decoder_regression_operation.set_param(json.dumps({**decoder_regression_param}))
            self.acl_decoder_regression_operation.set_weight(self.ascend_weight)

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
        q_lens = kwargs.get('q_lens', [])
        spec_mask = kwargs.get('spec_mask', None)

        input_length = len(input_ids)
        self.expert_array = self.placeholder
        if not self.enable_fused_routing:
            self.expert_array = torch.arange(self.config.num_experts_per_tok * input_length,
                                            dtype=torch.int32, device=input_ids.device)

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(max_seq_len if self.split_fuse_enable else self.max_base_len,
                                                          self.dtype, self.device)
            if self.split_fuse_enable and self.dtype == torch.bfloat16:
                atten_mask = atten_mask * -10000.0
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
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

            if self.split_fuse_enable:
                self.acl_encoder_operation_inputs.append(torch.tensor(q_lens).to(self.device).to(torch.int32))

            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            use_regression = False
            if self.prefix_cache_enable and q_lens == []:
                use_regression = True
                q_lens = []

            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            })
            if self.prefix_cache_enable and use_regression:
                self.acl_decoder_regression_operation_inputs = [
                    input_ids,
                    position_ids.to(torch.int64),
                    self.cos_embed,
                    self.sin_embed,
                    self.attn_mask_fake,
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
                    self.zero_hot,
                ]
                return self.acl_decoder_regression_operation_inputs, self.acl_param

            self.acl_decoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                spec_mask if self.speculate_enable or self.prefix_cache_enable else self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64) if self.prefix_cache_enable else self.lm_head_indices_fake,
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
            ]

            if self.split_fuse_enable or self.speculate_enable or self.prefix_cache_enable:
                self.acl_decoder_operation_inputs.append(torch.tensor(q_lens).to(self.device).to(torch.int32))

            return self.acl_decoder_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_param_dict = json.loads(acl_param)
            if self.prefix_cache_enable and acl_param_dict["qLen"] == []:
                model_operation = self.acl_decoder_regression_operation
            else:
                model_operation = self.acl_decoder_operation
            acl_model_out = model_operation.execute(acl_inputs, acl_param)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            msg = "Runtime Error, please refer to the logs for more info"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError(msg) from e
        return acl_hidden_state

    def init_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            if self.prefix_cache_enable:
                self.acl_decoder_regression_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])

if __name__ == "__main__":
    test_config = DeepseekConfig()
    test_weights = None
    model = FlashDeepseekForCausalLM(test_config, test_weights)