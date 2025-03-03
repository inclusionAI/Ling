# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch_npu
from atb_llm.utils.layers import (
    AttentionMask,
    TensorParallelHead,
)

from .config_gpt2 import Gpt2Config
from .modeling_gpt2 import FlashGPT2Model
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType

CPP_GPT2_MODEL_CLASS_NAME = "starcoder_DecoderModel"


class FlashGpt2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = config.num_attention_heads // self.tp_world_size  # 48
        
        # if num_key_value_heads is nondivisible 
        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
            // self.tp_world_size
        
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.quantize = config.quantize
        self.dtype = weights.dtype
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.device = weights.device
        # for ascend init
        self.init_ascend_operations(config)
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        
        self.max_seq_leneqlen_tensor = torch.tensor([0], dtype=torch.int)
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

        self.attention_mask_max_en = None
        self.attention_mask_max_de = None

        self.max_base_len = 128
        self.attn_mask = AttentionMask.static(self.max_base_len, dtype=self.dtype)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.acl_param = None
        self.model = FlashGPT2Model(config, weights)
        self.lm_head = TensorParallelHead.load(
            config,
            prefix="wte",
            weights=weights,
            is_norm=True
        )

        self.placeholder = torch.ones(1, dtype=self.dtype, device="npu")
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int).to(self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).to(self.device)
        self.attn_mask_fake = self.attn_mask \
            .get_attn_mask(1, dtype=self.dtype, device="cpu") \
            .to(self.device)

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  # position_ids
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        if lm_head_indices is not None:
            logits = logits[lm_head_indices]
        else:
            logits = logits

        return logits

    def init_ascend_operations(self, config: Gpt2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_GPT2_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_GPT2_MODEL_CLASS_NAME)

    def weight_format_cast(self, weight):
        if not self.soc_info.need_nz:
            return weight
        torch_npu.npu_format_cast_(weight, 29)
        return weight

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='self_attn',
            pack_name='qkv',
            sep_names=None,
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='up_proj',
            sep_names=None,
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.wte)
        weight_wrapper.register_embedding(self.model.wpe)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return (weight_wrapper)

    def init_ascend_weight(self):
        weight = self.get_weights()
        self.ascend_weight, linear_types, pack_quant_configs, linear_transpose_types =\
            weight.weights, weight.linear_type, weight.pack_quant_type, weight.linear_transpose_types
        
        linear_transpose_types = []
        for _ in range(self.num_layers):
            linear_transpose_types.append([0, -1, -1, 0, 0, -1, 0])

        coder_param = {
            "isFA": False,
            "isUnpadInputs": True,
            "normEps": self.config.layer_norm_epsilon,
            "positionEmbeddingType": PositionEmbeddingType.ABSOLUTE,
            "normType": NormType.LAYER_NORM,
            "isBF16": self.dtype == torch.bfloat16,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": False,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "LayerNormEps": self.layer_norm_epsilon,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.num_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "layerNormEps": self.layer_norm_epsilon,
            "headNum": self.num_heads,
            "dk": self.head_dim,
            "kvHead": 1,
            "layerNum": self.num_layers,
            "rankSize": self.tp_world_size,
            "numHeadsPerPartition": self.num_key_value_heads,
        }
        encoder_param = {**coder_param, "isPrefill": True}
        decoder_param = {**coder_param, "isPrefill": False}
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
                                  lm_head_indices: Optional[torch.Tensor] = None):

        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                    kv_cache[0][0].device)
            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.placeholder,
                self.placeholder,
                atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64),
            ]
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.placeholder,
                self.placeholder,
                self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                self.lm_head_indices_fake,
            ]
            return self.acl_decoder_operation_inputs, self.acl_param
