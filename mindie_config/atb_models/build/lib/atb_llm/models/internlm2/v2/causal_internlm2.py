# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional
import torch

from atb_llm.utils.layers import load_column_multi
from ..base.causal_lm import CausalLM
from .modeling_internlm2 import FlashInternlm2Model
from .config_internlm2 import Internlm2Config

from ...utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper


class Internlm2ForCausalLM(CausalLM):
    def __init__(self, config, weights, **kwargs):
        self.ascend_weight = None
        self.linear_type = None
        self.pack_quant_config = None
        self.acl_param = None
        self.sin_embed = None
        self.cos_embed = None
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights, **kwargs)
        self.model = FlashInternlm2Model(config, weights)

        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config
        self.placeholder = torch.zeros(1, dtype=self.dtype).npu()
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

        self.in_tensor_length = 12
        self.acl_encoder_operation_inputs: list = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list = [None] * self.in_tensor_length

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

    def init_ascend_operations(self, config: Internlm2Config):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("internlm2_20b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("internlm2_20b_DecoderModel")

    def get_weights(self):
        attn_module_names = AttnModuleNames(
            norm_name='input_layernorm',
            pack_name='self_attn.query_key_value',
            q_name='self_attn.q_proj',
            k_name='self_attn.k_proj',
            v_name='self_attn.v_proj',
            o_name='self_attn.o_proj'
        )
        mlp_module_names = MlpModuleNames(
            norm_name='post_attention_layernorm',
            pack_name='mlp.gate_up_proj',
            gate_name='mlp.gate_proj',
            up_name='mlp.up_proj',
            down_name='mlp.down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_module_names, mlp_module_names)
        weight_wrapper.register_embedding(self.model.state_dict(), 'embed_tokens')
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            layer_dict = layer.state_dict()
            weight_wrapper.register_layer(layer_dict, layer.self_attn.pack_type, layer.mlp.pack_type, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.state_dict(), 'norm')
        weight_wrapper.register_model_lmhead(self.state_dict(), 'lm_head')
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        self.linear_type = weight_wrapper.linear_type
        self.pack_quant_config = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        coder_param = {
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": True,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": self.pack_quant_config,
            "linearQuantType": self.linear_type,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
        }
        encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": self.lcoc_enable}
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_kvcache(self, input_ids, past_key_value):
        super().init_kvcache(input_ids, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[bool],
                                  max_seq_len: int,
                                  ):
        self.ascend_rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                                self.device,
                                                                max_seq_len)
        self.cos_embed = self.ascend_rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.ascend_rotary_embedding.get_sin_cached_total()

        if cu_seqlen_prefill:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids.shape[1]] * self.batch_num
            })
            self.acl_encoder_operation_inputs[0] = input_ids
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
            self.acl_decoder_operation_inputs[0] = input_ids
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
