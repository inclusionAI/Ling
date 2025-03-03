#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
from typing import Optional, List, Tuple

import torch

from .modeling_minicpm import FlashMiniCpmModel, MiniCpmConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers import load_column_multi
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.env import ENV

CPP_MINICPM_MODEL_CLASS_NAME = "minicpm_MiniCPMDecoderModel"


class FlashMinicpmForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, lmhead_prefix="model.embed_tokens", model_prefix="model"):
        super().__init__(config, weights)

        self.model = FlashMiniCpmModel(config, weights, model_prefix)

        self.lm_head = load_column_multi(
            config,
            prefixes=[lmhead_prefix],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.in_tensor_length = 12
        self.total_head_nums = config.hidden_size // self.head_dim
        self.acl_encoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length

        self.position_embedding_type = config.pe_type
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.scale_emb = config.scale_emb
        self.scale_depth = config.scale_depth
        self.dim_model_base = config.dim_model_base
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")
        self.acl_param = None
        self.ascend_weight = None
        self.cos_embed = None
        self.sin_embed = None
        self.atten_mask_cpu = None
        self.skip_word_embedding = False
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: MiniCpmConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_MINICPM_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_MINICPM_MODEL_CLASS_NAME)

    def get_weights(self):
        attn_wrapper, mlp_wrapper = self.get_attn_mlp_wrapper()
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.mlp
                del layer.post_attention_layernorm
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def get_attn_mlp_wrapper(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        return attn_wrapper, mlp_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        if self.position_embedding_type == "ROPE":
            position_embedding_type = PositionEmbeddingType.ROPE
        else:
            position_embedding_type = PositionEmbeddingType.ALIBI
        linear_transpose_types, linear_types, pack_quant_configs = self.get_trans_info(weight_wrapper)

        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,            
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "hiddenSize": self.hidden_size,
            "rankTableFile": ENV.rank_table_file,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,        
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": False,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "positionEmbeddingType": position_embedding_type,
            "enableAddNorm": False,                   
            "enableCompressHead": self.compress_head_enable,
            "numHiddenLayers": self.config.num_hidden_layers,
            "scale_emb": self.scale_emb,
            "scale_depth": self.scale_depth,
            "dim_model_base": self.dim_model_base,
            "num_hidden_layers": self.num_hidden_layers,
            "enableSpeculate": False,
        }

        encoder_param = {
            **coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable,
            "skipWordEmbedding": self.skip_word_embedding
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False
        }

        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def get_trans_info(self, weight_wrapper):
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        return linear_transpose_types, linear_types, pack_quant_configs

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):      
        self._init_rope_cos_sin(max_seq_len, dtype, device)

    def get_attention_mask(self, max_base_len, dtype, device):
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (max_base_len, max_base_len), fill_value=min_dtype, dtype=dtype, device=device
        )
        if max_base_len != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        return causal_mask    

    def prepare_inputs_for_ascend(
            self, input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype,
                                                      self.device)
            self.init_cos_sin_table(self.max_position_embeddings, self.head_dim, self.dtype, self.device)

            if self.soc_info.need_nz:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                               dtype=torch.int64, device=input_ids.device)


            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = self.cos_embed
            self.acl_encoder_operation_inputs[3] = self.sin_embed
            self.acl_encoder_operation_inputs[4] = attention_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)

            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })

            #IN_TENSOR_KV_CACHE_IDX
            self.acl_encoder_operation_inputs[7] = self.placeholder

            #IN_TENSOR_TOKEN_OFFSET
            self.acl_encoder_operation_inputs[8] = self.placeholder

            #IN_TENSOR_PLACE_HOLDER
            self.acl_encoder_operation_inputs[9] = self.placeholder

            #IN_TENSOR_SEQ_LEN
            self.acl_encoder_operation_inputs[10] = input_lengths.to(torch.int32)

            #IN_TENSOR_LOGTIS_INDICES
            self.acl_encoder_operation_inputs[11] = lm_head_indices.to(torch.int64)

            return self.acl_encoder_operation_inputs, self.acl_param
        else:

            spec_mask = kwargs.get('spec_mask', None)
            attention_mask = spec_mask if self.speculate_enable else self.attn_mask_fake

            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.cos_embed
            self.acl_decoder_operation_inputs[3] = self.sin_embed
            self.acl_decoder_operation_inputs[4] = attention_mask
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = self.placeholder
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[11] = self.lm_head_indices_fake

            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": kwargs.get('q_lens', [])
            })

            return self.acl_decoder_operation_inputs, self.acl_param

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        self.rotary_embedding.update_cos_sin_cache_total(dtype, device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
