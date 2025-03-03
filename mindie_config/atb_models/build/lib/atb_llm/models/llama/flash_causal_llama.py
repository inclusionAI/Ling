# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from typing import Optional, List, Tuple
import json
import math

import torch
import torch_npu

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.quantize.pack_type import QuantType
from .config_llama import LlamaConfig
from .modeling_llama import FlashLlamaModel
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.env import ENV
from ...utils.layers import load_column_multi, TensorHead, TensorParallelHead
from ...utils.layers.embedding.cos_sin_table import CosSinTable
from ...utils.log import logger, print_log
from ...utils.log.error_code import ErrorCode
from ...utils.adapter_manager import AdapterIdsType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.layers.norm.fast_layer_norm import NormType


CPP_LLAMA_MODEL_CLASS_NAME = "llama_LlamaDecoderModel"


class FlashLlamaForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, lmhead_prefix="lm_head", model_prefix="model", **kwargs):
        super().__init__(config, weights, **kwargs)

        self.model = FlashLlamaModel(config, weights, model_prefix, attn_decode_backend=self.attn_decode_backend)
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=lmhead_prefix,
                weights=weights,
                is_norm=False,
            )
        elif config.tie_word_embeddings:
            self.lm_head = TensorParallelHead.load(
                config,
                prefix="model.embed_tokens",
                weights=weights,
                is_norm=True,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=[lmhead_prefix],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.total_head_nums = config.hidden_size // self.head_dim

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.position_embedding_type = config.pe_type
        self.alibi_bias_max = config.alibi_bias_max
        self.acl_param = None
        self.ascend_weight = None
        self.atten_mask_cpu = None
        self.alibi_mask_compress = True
        self.skip_word_embedding = False
        if self.position_embedding_type != "ROPE" and self.position_embedding_type != "ALIBI":
            error_msg = "`pe_type` is only support for type: `ROPE` and `ALIBI`, loaded from config.json -> pe_type."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(error_msg)
        self.cos_embed = None
        self.sin_embed = None
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None

        self.mc2_enable = False
        self.soc_info.matmul_nd_nz = ENV.matmul_nd_nz_enable and self.soc_info.soc_version == 225 \
            and not self.mc2_enable and ((config.quantize is None) or (config.quantize == QuantType.FLOAT))

        self.cos_sin_table_params = CosSinTable()
        self.cos_sin_table_params.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.cos_sin_table_params.rope_vanilla_theta = config.rope_vanilla_theta
        self.cos_sin_table_params.rope_mscale = config.rope_mscale
        self.cos_sin_table_params.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.cos_sin_table_params.rope_theta = self.rope_theta

        self.acl_encoder_operation_inputs: list[None | torch.Tensor] = \
            [None] * self.get_in_tensor_size(encoder=True)
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] = \
            [None] * self.get_in_tensor_size(encoder=False)
        if self.prefix_cache_enable:
            self.acl_decoder_regression_operation_inputs: list[None | torch.Tensor] = \
                [None] * self.get_in_tensor_size(encoder=False, regression=True)
        self.acl_decoder_regression_operation = None

        self.acl_multi_lora_encoder_operation = None
        self.acl_multi_lora_decoder_operation = None

        self.acl_base_encoder_operation = None
        self.acl_base_decoder_operation = None

        self.decode_pffset_index = None
        self.in_reshape_seqlen = None
        self.block_nums_list = None
        self.razor_offset = None
        self.in_ra_seqlens = None
        self.pffset_index = None

        self.warmup = True

    def get_in_tensor_size(self, encoder=True, regression=False):
        base_size = 13
        if regression:
            return base_size
        if self.compress_head_enable:
            base_size += 5  # batch_wins, input_length_js
        if not encoder and self.speculate_enable:
            base_size += 1  # 1: q_len
        if encoder and self.split_fuse_enable:
            base_size += 1  # 1: q_len
        if ENV.long_seq_enable:
            base_size += 3
        return base_size

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: LlamaConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)

    def get_weights(self):
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
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
            if self.config.quantization_config.reduce_quant_type is not None:
                weight_wrapper.register_layer_reducequant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        rank_table_file = ENV.rank_table_file
        if self.position_embedding_type == "ROPE":
            position_embedding_type = PositionEmbeddingType.ROPE
        else:
            position_embedding_type = PositionEmbeddingType.ALIBI
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "enableFA3": self.config.quantization_config.fa_quant_type is not None,
            "enableReduceQuant": self.config.quantization_config.reduce_quant_type is not None,
            "attnBackend": self.attn_decode_backend,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": position_embedding_type,
            "enableAddNorm": False,
            "enableCompressHead": self.compress_head_enable,
            "enableLora": self.adapter_manager is not None,
            "quantGroupSize": self.config.quantization_config.group_size,
            "isLongSeq": ENV.long_seq_enable,
        }
        if self.config.model_type == "zhinao":
            coder_param.update({"linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers})
            coder_param.update({"splitWithStride": True})
        encoder_param = {
            **coder_param, "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "enableSpeculate": False,
            "skipWordEmbedding": self.skip_word_embedding,
            "enableSplitFuse": self.split_fuse_enable
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False,
            "enableSpeculate": self.speculate_enable,
            "enablePrefixCache": self.prefix_cache_enable
        }
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)
        
        if self.prefix_cache_enable:
            self.init_prefix_cache_regression_weight(coder_param)

        if self.adapter_manager is not None:
            self.acl_multi_lora_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)
            self.acl_multi_lora_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)

            encoder_param.update({"loraEnableGMM": True})
            self.acl_multi_lora_encoder_operation.set_param(json.dumps({**encoder_param}))
            decoder_param.update({"loraEnableGMM": True})
            self.acl_multi_lora_decoder_operation.set_param(json.dumps({**decoder_param}))

            self.acl_multi_lora_encoder_operation.set_weight(self.ascend_weight)
            self.acl_multi_lora_decoder_operation.set_weight(self.ascend_weight)

            self.acl_base_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)
            self.acl_base_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)

            disable_support_lora = {"enableLora": False}
            encoder_param.update(disable_support_lora)
            self.acl_base_encoder_operation.set_param(json.dumps({**encoder_param}))
            decoder_param.update(disable_support_lora)
            self.acl_base_decoder_operation.set_param(json.dumps({**decoder_param}))

            self.acl_base_encoder_operation.set_weight(self.ascend_weight)
            self.acl_base_decoder_operation.set_weight(self.ascend_weight)

    def init_prefix_cache_regression_weight(self, coder_param):
        # prefix cache特性多加一张图用于自回归decode
        self.acl_decoder_regression_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)
        decoder_regression_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False,
            "enableSpeculate": False
        }
        self.acl_decoder_regression_operation.set_param(json.dumps({**decoder_regression_param}))
        self.acl_decoder_regression_operation.set_weight(self.ascend_weight)

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        if self.cos_sin_table_params.rope_given_inv_feq_str is None \
            and self.cos_sin_table_params.rope_vanilla_theta is None:
            self._init_rope_cos_sin(max_seq_len, dtype, device)
        else:
            self.cos_sin_table_params.dim = dim
            self.cos_sin_table_params.offset = 0
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len, dtype, device, self.cos_sin_table_params
                )

    def razor_attention_input(self,
                           input_lengths: torch.Tensor,
                           kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                           max_seq_len: int,
                           max_out_len: int):
        layer_nums = self.config.num_hidden_layers
        head_nums = self.num_key_value_heads
        batch_size = input_lengths.shape[0]
        block_size = kv_cache[0][0].shape[1]

        block_nums_list = [i[0].shape[0] * block_size for i in kv_cache]
        block_nums = int(max(block_nums_list) / block_size)
        self.block_nums_list = block_nums_list
        npu = "npu"
        # 1.根据离线校准获得的 head_dict 构造 windows
        wins = torch.zeros((batch_size, head_nums), dtype=torch.int32).npu()
        wins_keep = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32).npu()
        wins_drop = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32).npu()
        pffset_index = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32).npu()

        # wins for llama3.1-70B
        head_dict = {
            'prefix_matching': {0:[0, 1, 2, 3, 4, 5, 6, 7], 1:[0, 1, 2, 3, 4, 5, 6, 7],
                    18: [5, 0, 2, 6, 1, 7], 76: [7, 1, 5, 3], 35: [5, 3, 2, 0, 6, 4],
                    74: [0, 3, 2, 1, 5], 52: [2, 7, 4, 6, 3], 56: [0, 5, 2, 3, 4],
                    77: [6, 0, 5, 1, 7, 4], 64: [3, 6], 33: [2, 1, 5, 4], 53: [0, 2],
                    37: [3, 4, 1, 2, 0, 5], 54: [1, 3], 21: [4, 6, 7, 3, 2], 47: [2],
                    72: [4], 31: [2, 4, 5, 3, 0, 6], 44: [5, 0], 67: [7, 6],
                    22: [1, 6, 4, 3], 68: [6, 2, 1, 7], 23: [5, 1, 4, 2, 6],
                    71: [0, 6, 4, 1], 39: [5, 1, 4], 36: [6, 5, 1, 2, 0],
                    27: [4, 2, 6, 5, 1], 73: [0, 4, 7, 5, 2, 1], 30: [6, 4, 5, 3],
                    14: [4, 2], 38: [6, 2, 3, 7], 60: [2, 3, 7, 6], 34: [0, 3, 4, 5],
                    41: [7, 2, 4], 19: [4, 0], 69: [4, 5], 29: [1, 6, 3, 7],
                    75: [2, 3, 0, 4, 5, 1], 61: [0, 4], 49: [0],
                    25: [1, 7, 4], 57: [0, 1, 3, 6], 17: [6, 3, 7, 1, 2], 58: [0],
                    24: [0, 5, 3], 32: [2, 5, 6], 42: [3], 55: [0], 70: [1, 6], 28: [5, 1, 0],
                    48: [7, 6], 50: [4], 16: [4, 5], 7: [3, 0], 63: [7, 4], 51: [3], 78: [0],
                    5: [5], 59: [4], 26: [3, 4, 5], 66: [0], 15: [2], 40: [4], 43: [2],
                    45: [0], 6: [4]},
            'copying': {14: [2], 17: [7], 74: [0], 33: [2], 52: [2], 15: [2],
                        31: [4], 38: [6], 71: [0], 23: [5], 27: [4, 2], 30: [4, 6],
                        19: [0], 77: [5], 75: [3], 47: [2], 21: [6]}
        }
        inductive_head = head_dict["prefix_matching"]
        copying_head = head_dict["copying"]
        kv_tp_size = self.tp_world_size
        for batch_idx in range(batch_size):
            first_sink = 40
            last_sink = max(4000, input_lengths[batch_idx] // 5)

            if input_lengths[batch_idx] - first_sink - last_sink - 1 <= 0:  # 不需要压缩
                wins[batch_idx][:] = 0
            else:  # 需要压缩
                wins[batch_idx][:] = input_lengths[batch_idx] - first_sink - last_sink
            for layer_idx in range(layer_nums):
                for head_idx in range(head_nums):
                    cur_head_idx = head_idx + self.tp_rank * kv_tp_size // self.tp_world_size * head_nums
                    is_inductive_head = layer_idx in inductive_head and cur_head_idx in inductive_head.get(layer_idx)
                    is_copying_head = layer_idx in copying_head and cur_head_idx in copying_head.get(layer_idx)
                    # 不需要压缩的head
                    if (is_inductive_head or is_copying_head) or\
                            (input_lengths[batch_idx] - first_sink - last_sink - 1 <= 0):
                        wins_drop[batch_idx][layer_idx][head_idx] = 0
                        wins_keep[batch_idx][layer_idx][head_idx] = input_lengths[batch_idx]
                        pffset_index[batch_idx][layer_idx][head_idx] = -1
                    # 需要压缩的head
                    else:
                        wins_drop[batch_idx][layer_idx][head_idx] = \
                            input_lengths[batch_idx] - first_sink - last_sink
                        wins_keep[batch_idx][layer_idx][head_idx] = first_sink + 1 + last_sink
                        pffset_index[batch_idx][layer_idx][head_idx] = first_sink

        # 2.重新定义 block_tables
        if block_size != 0:
            max_need_blocks = math.ceil((max_seq_len + max_out_len) / block_size)
        else:
            max_need_blocks = 0
        block_tables = torch.zeros((batch_size, layer_nums, head_nums, max_need_blocks),
                                   dtype=torch.int32, device=npu)

        cur_need_blocks = torch.ceil((wins_keep.float() + max_out_len) / block_size).to(torch.int32)
        block_indices = (torch.arange(max_need_blocks, dtype=torch.int32, device=npu).
                         expand(batch_size, layer_nums, head_nums, max_need_blocks))
        global_offsets = torch.cumsum(cur_need_blocks, dim=-1, dtype=torch.int32) - cur_need_blocks
        valid_mask = block_indices < cur_need_blocks.unsqueeze(-1)
        broadcasted_block_indices = block_indices + global_offsets.unsqueeze(-1)
        valid_mask_indices = valid_mask.nonzero(as_tuple=True)
        block_tables[valid_mask_indices] = broadcasted_block_indices[valid_mask]
    
        # 3.重新定义 slots
        self.decoder_slots = torch.zeros((batch_size, layer_nums, head_nums), dtype=torch.int32, device=npu)
        offsets = (block_tables[:, :, :, 0] * block_size).to(torch.int32)
        seq_lens = wins_keep
        slots = offsets
        self.decoder_slots = offsets + seq_lens - 1
        # 4.定义 PageAttention 所需输入 ra_offset
        ra_offset = torch.zeros((layer_nums, block_nums * block_size), dtype=torch.float32, device=npu)
        mask = wins_drop > 0
        log_wins_drop = torch.log(wins_drop)
        valid_offsets = offsets + first_sink
        layer_indices = (torch.arange(layer_nums, dtype=torch.int32, device=npu).unsqueeze(0).unsqueeze(2).
                         expand(batch_size, layer_nums, head_nums))
        valid_offsets_flat = valid_offsets[mask]
        layer_indices_flat = layer_indices[mask]
        ra_offset.index_put_((layer_indices_flat, valid_offsets_flat), log_wins_drop[mask], accumulate=False)

        # reshape 成需要的维度
        in_ra_seqlens = wins_keep.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        block_tables = block_tables.transpose(0, 1).reshape(layer_nums, batch_size * head_nums, max_need_blocks)
        slots = slots.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        pffset_index = pffset_index.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        ra_offset = ra_offset.reshape(layer_nums, block_nums, block_size)

        self.decoder_slots = self.decoder_slots.transpose(0, 1).reshape(layer_nums, batch_size * head_nums)
        self.block_tables_global = block_tables

        self.wins_global = wins.reshape(batch_size * head_nums)
        self.razor_offset = ra_offset
        self.in_ra_seqlens = in_ra_seqlens
        self.pffset_index = pffset_index

        self.decode_pffset_index = torch.full((layer_nums, batch_size * head_nums), -1, dtype=torch.int32, device=npu)
        return block_tables, slots

    def get_ntk_alpha(self, seq_len):
        ntk_alpha = (self.scaling_factor * seq_len / self.config.max_position_embeddings) - (self.scaling_factor - 1)
        return ntk_alpha

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
        if self.adapter_manager is not None:
            # 更新adapter
            adapter_ids = kwargs.get("adapter_ids")
            effective_adapter_ids = self.process_adapter_ids(adapter_ids)
            adapter_weights = self.prepare_adapter_weights(effective_adapter_ids)
            if adapter_weights:
                self.update_adapter_weights(adapter_weights, self.acl_encoder_operation_inputs,
                                            self.get_in_tensor_size(encoder=True))
                self.update_adapter_weights(adapter_weights, self.acl_decoder_operation_inputs,
                                            self.get_in_tensor_size(encoder=False))

        n_head = self.num_attention_heads
        q_lens = kwargs.get('q_lens', [])
        spec_mask = kwargs.get('spec_mask', None)
        atten_mask = kwargs.get('atten_mask', None)

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                            dtype=torch.int64, device=input_ids.device)

        if ENV.long_seq_enable:
            ntk_alpha = self.get_ntk_alpha(max_seq_len)
            self.rotary_embedding.dynamic_ntk_inv_freq_llama(self.config, max_seq_len, self.device, ntk_alpha)
        position_ids_expanded = self.rotary_embedding.position_ids_expanded
        inv_freqs = self.rotary_embedding.ntk_inv_freqs
        pos_lens = self.rotary_embedding.pos_lens

        if self.cos_embed is None and self.sin_embed is None:
            if self.position_embedding_type == "ROPE":
                self.init_cos_sin_table(self.max_position_embeddings, self.head_dim, self.dtype, self.device)
            elif self.position_embedding_type == "ALIBI":
                self.cos_embed = self.placeholder
                self.sin_embed = self.placeholder
            else:
                logger.error("error: `pe_type` is only support for type: `ROPE` and `ALIBI`, \
                             loaded from config.json -> pe_type.", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)

        if is_prefill:
            if self.position_embedding_type == "ROPE":
                atten_mask = self.attn_mask.get_attn_mask(
                    max_seq_len if self.split_fuse_enable else self.max_base_len, self.dtype, self.device)
                # BF16 PA算子需要使用-10000.0表示mask掉的部分
                if self.split_fuse_enable and self.dtype == torch.bfloat16:
                    atten_mask = atten_mask * -10000.0
            elif self.position_embedding_type == "ALIBI":
                if self.atten_mask_cpu is None:
                    self.atten_mask_cpu = self._gen_alibi_mask(self.total_head_nums, self.max_position_embeddings,
                                                               self.alibi_bias_max)[
                                          self.tp_rank * n_head:(self.tp_rank + 1) * n_head, :, :].to(self.dtype)
                if self.alibi_mask_compress:
                    # 算子要求: 小于128则按实际长度切，大于128则按128切，算子内部扩展到实际长度
                    slice_len = max_seq_len if max_seq_len <= 128 else 128
                    atten_mask = self.atten_mask_cpu[:, :, :slice_len].npu()
                else:
                    atten_mask = self.atten_mask_cpu[:, :max_seq_len, :max_seq_len].npu()
            else:
                logger.error("error: `pe_type` is only support for type: `ROPE` and `ALIBI`, \
                             loaded from config.json -> pe_type.", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)

            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            })
            if self.compress_head_enable:
                max_out_len = kwargs.get('max_out_len', 256)
                if max_out_len <= 0:
                    error_msg = "error: max_out_len must be greater than 0."
                    logger.error(error_msg, ErrorCode.ATB_MODELS_INTERNAL_ERROR)
                    raise ValueError(error_msg)
                block_tables, slots = self.razor_attention_input(input_lengths, kv_cache, max_seq_len, max_out_len)
                block_tables = block_tables.reshape(-1, block_tables.shape[-1])
                slots = slots.reshape(-1)
                self.acl_param = json.dumps({
                    "seqLen": input_lengths.tolist(),
                    "qLen": q_lens,
                    "blockNumsList": self.block_nums_list
                })

            self.acl_encoder_operation_inputs[0] = self.placeholder if self.skip_word_embedding else input_ids
            self.acl_encoder_operation_inputs[1] = input_ids if self.skip_word_embedding else self.placeholder
            self.acl_encoder_operation_inputs[2] = position_ids.to(torch.int32)
            self.acl_encoder_operation_inputs[3] = self.placeholder if ENV.long_seq_enable else self.cos_embed
            self.acl_encoder_operation_inputs[4] = self.placeholder if ENV.long_seq_enable else self.sin_embed
            self.acl_encoder_operation_inputs[5] = atten_mask
            self.acl_encoder_operation_inputs[6] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[8] = self.placeholder
            self.acl_encoder_operation_inputs[9] = self.placeholder
            self.acl_encoder_operation_inputs[10] = self.placeholder
            self.acl_encoder_operation_inputs[11] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[12] = lm_head_indices.to(torch.int64)
            if self.compress_head_enable:
                self.in_reshape_seqlen = input_lengths.to(torch.int32)
                self.acl_encoder_operation_inputs[13] = self.wins_global
                self.acl_encoder_operation_inputs[14] = self.in_ra_seqlens.reshape(-1)
                self.acl_encoder_operation_inputs[15] = self.pffset_index.reshape(-1)
                self.acl_encoder_operation_inputs[16] = self.razor_offset.reshape(-1, self.razor_offset.shape[-1])
                self.acl_encoder_operation_inputs[17] = self.in_reshape_seqlen
            if self.split_fuse_enable:
                self.acl_encoder_operation_inputs[13] = torch.tensor(q_lens).to(self.device).to(torch.int32)
            if self.adapter_manager is not None and effective_adapter_ids != ["base"]:
                self.acl_encoder_operation_inputs[13] = self.calculate_adapter_group_size(
                    effective_adapter_ids, self.acl_encoder_operation_inputs[11], is_prefill=True)
            if ENV.long_seq_enable:
                self.acl_encoder_operation_inputs[13] = position_ids_expanded
                self.acl_encoder_operation_inputs[14] = inv_freqs
                self.acl_encoder_operation_inputs[15] = pos_lens

            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            use_regression = False
            if self.prefix_cache_enable and q_lens == []:  # 开启prefix cache时q_lens为空时使用自回归
                use_regression = True

            self.acl_param = json.dumps({
                "seqLen": self.in_ra_seqlens.reshape(-1).tolist() \
                    if self.compress_head_enable else input_lengths.tolist(),
                "qLen": q_lens
            })
            if self.speculate_enable and self.soc_info.need_nz and not use_regression:
                spec_mask = self.transdata_operation.execute([spec_mask])[0]

            if self.position_embedding_type == "ROPE":
                if self.speculate_enable:
                    atten_mask = self.attn_mask_fake if use_regression else spec_mask
                else:
                    atten_mask = self.attn_mask_fake
            elif self.position_embedding_type == "ALIBI":
                atten_mask = self._gen_alibi_mask_decoder(self.total_head_nums, position_ids.tolist(),
                                        max_seq_len, self.alibi_bias_max)[:,
                                        self.tp_rank * n_head:(self.tp_rank + 1) * n_head, :, :].to(self.dtype).npu()
            else:
                logger.error("error: position_embedding_type is inllegal", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            if self.compress_head_enable:
                self.in_ra_seqlens = self.in_ra_seqlens + 1
                self.decoder_slots = self.decoder_slots + 1
                slots = self.decoder_slots
                block_tables = self.block_tables_global
                block_tables = block_tables.reshape(-1, block_tables.shape[-1])
                slots = slots.reshape(-1)
                self.acl_param = json.dumps({
                    "seqLen": self.in_ra_seqlens.reshape(-1).tolist(),
                    "qLen": q_lens,
                    "blockNumsList": self.block_nums_list
                })

            if self.prefix_cache_enable and use_regression:  # 自回归decode
                self.acl_decoder_regression_operation_inputs[0] = input_ids
                self.acl_decoder_regression_operation_inputs[1] = self.placeholder
                self.acl_decoder_regression_operation_inputs[2] = position_ids.to(torch.int64)
                self.acl_decoder_regression_operation_inputs[3] = self.cos_embed
                self.acl_decoder_regression_operation_inputs[4] = self.sin_embed
                self.acl_decoder_regression_operation_inputs[5] = atten_mask
                self.acl_decoder_regression_operation_inputs[6] = block_tables.to(torch.int32)
                self.acl_decoder_regression_operation_inputs[7] = slots.to(torch.int32)
                self.acl_decoder_regression_operation_inputs[8] = self.placeholder
                self.acl_decoder_regression_operation_inputs[9] = self.placeholder
                self.acl_decoder_regression_operation_inputs[10] = self.placeholder
                self.acl_decoder_regression_operation_inputs[11] = input_lengths.to(torch.int32)
                self.acl_decoder_regression_operation_inputs[12] = self.lm_head_indices_fake
                return self.acl_decoder_regression_operation_inputs, self.acl_param

            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = self.placeholder
            self.acl_decoder_operation_inputs[2] = position_ids.to(torch.int32)
            self.acl_decoder_operation_inputs[3] = self.placeholder if ENV.long_seq_enable else self.cos_embed
            self.acl_decoder_operation_inputs[4] = self.placeholder if ENV.long_seq_enable else self.sin_embed
            self.acl_decoder_operation_inputs[5] = atten_mask
            self.acl_decoder_operation_inputs[6] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = self.placeholder
            self.acl_decoder_operation_inputs[11] = input_lengths.to(torch.int32)
            if self.prefix_cache_enable:
                self.acl_decoder_operation_inputs[12] = lm_head_indices.to(torch.int64)
            else:
                self.acl_decoder_operation_inputs[12] = self.lm_head_indices_fake
            if self.compress_head_enable:
                self.in_reshape_seqlen = torch.ones(input_lengths.shape[0], dtype=torch.int32).npu()
                self.acl_decoder_operation_inputs[13] = self.wins_global
                self.acl_decoder_operation_inputs[14] = self.in_ra_seqlens.reshape(-1)
                self.acl_decoder_operation_inputs[15] = self.decode_pffset_index.reshape(-1)
                self.acl_decoder_operation_inputs[16] = self.razor_offset.reshape(-1, self.razor_offset.shape[-1])
                self.acl_decoder_operation_inputs[17] = self.in_reshape_seqlen
            if self.speculate_enable:
                self.acl_decoder_operation_inputs[13] = torch.tensor(q_lens).to(self.device).to(torch.int32)
            if self.adapter_manager is not None and effective_adapter_ids != ["base"]:
                self.acl_decoder_operation_inputs[13] = self.calculate_adapter_group_size(
                    effective_adapter_ids,
                    torch.ones_like(input_ids, device=self.device, dtype=torch.int64),
                    is_prefill=False)
            if ENV.long_seq_enable:
                self.acl_decoder_operation_inputs[13] = position_ids_expanded
                self.acl_decoder_operation_inputs[14] = inv_freqs
                self.acl_decoder_operation_inputs[15] = pos_lens
            return self.acl_decoder_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        has_multiple_adapter_ids = self.adapter_manager is not None \
            and self.adapter_manager.previous_adapter_ids.record_type != AdapterIdsType.SINGLE
        only_base_adapter = self.adapter_manager is not None \
            and self.adapter_manager.previous_adapter_ids.adapter_ids == ["base"]


        if is_prefill:
            if has_multiple_adapter_ids:
                model_operation = self.acl_multi_lora_encoder_operation
                acl_model_out = model_operation.execute(acl_inputs, acl_param)
            elif only_base_adapter:
                model_operation = self.acl_base_encoder_operation
                acl_model_out = model_operation.execute(acl_inputs[:13], acl_param)
            else:
                model_operation = self.acl_encoder_operation
                acl_model_out = model_operation.execute(acl_inputs, acl_param)
        else:
            if has_multiple_adapter_ids:
                model_operation = self.acl_multi_lora_decoder_operation
                acl_model_out = model_operation.execute(acl_inputs, acl_param)
            elif only_base_adapter:
                model_operation = self.acl_base_decoder_operation
                acl_model_out = model_operation.execute(acl_inputs[:13], acl_param)
            elif self.prefix_cache_enable and \
                    len(acl_inputs) == self.get_in_tensor_size(encoder=False, regression=True):
                model_operation = self.acl_decoder_regression_operation  # prefix cache自回归decode
                acl_model_out = model_operation.execute(acl_inputs, acl_param)
            else:
                model_operation = self.acl_decoder_operation
                acl_model_out = model_operation.execute(acl_inputs, acl_param)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            logger.error("运行时报错，请开启日志进一步定位问题", 
                         ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        if self.warmup and ENV.long_seq_enable:
            self.rotary_embedding.set_ntk_cache(self.config.max_position_embeddings, 
                                                self.rotary_embedding.inv_freq, self.device)
            self.warmup = False
        return acl_hidden_state

    def init_kvcache(self, kv_cache):
        kcache_id_diff = self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id_diff = self.ascend_vcache_id != id(kv_cache[0][1])
        kcache_shape_diff = self.ascend_kcache_shape != kv_cache[0][0].shape
        vcache_shape_diff = self.ascend_vcache_shape != kv_cache[0][1].shape
        kcache_diff = not self.ascend_kcache_id or kcache_id_diff or kcache_shape_diff
        vcache_diff = not self.ascend_vcache_id or vcache_id_diff or vcache_shape_diff
        if kcache_diff or vcache_diff:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            print_log(self.tp_rank, logger.info, f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            if self.prefix_cache_enable:
                self.acl_decoder_regression_operation.set_kv_cache(k_caches, v_caches)
            if self.acl_multi_lora_encoder_operation is not None:
                self.acl_multi_lora_encoder_operation.set_kv_cache(k_caches, v_caches)
            if self.acl_multi_lora_decoder_operation is not None:
                self.acl_multi_lora_decoder_operation.set_kv_cache(k_caches, v_caches)
            if self.acl_base_encoder_operation is not None:
                self.acl_base_encoder_operation.set_kv_cache(k_caches, v_caches)
            if self.acl_base_decoder_operation is not None:
                self.acl_base_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def _get_interleave(self, n, alibi_bias_max=8.0):
        def _get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (0.5 ** (alibi_bias_max / n))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self._get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def _fill_with_neg_inf(self, t):
        return t.float().fill_(float("-inf")).type_as(t)

    def _gen_alibi_mask(self, n_head, max_pos, alibi_bias_max=8.0):
        slopes = torch.Tensor(self._get_interleave(n_head, alibi_bias_max))
        tensor_list = []
        # 算子要求的压缩alibi mask shape为 [head_num, max_seq, 128]
        for i in range(128):
            tensor = torch.empty(max_pos).fill_(-float('inf'))
            tensor[i:] = -1 * torch.arange(0, max_pos - i)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0).t()
        tensor = tensor.expand(n_head, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        return alibi_mask

    def _gen_alibi_mask_decoder(self, n_head, pos_list, max_pos, alibi_bias_max=8.0):
        slopes = torch.Tensor(self._get_interleave(n_head, alibi_bias_max))
        tensor_list = []
        for pos in pos_list:
            tensor = torch.empty(max_pos).fill_(-float('inf'))
            tensor[:pos + 1] = torch.arange(-pos, 1)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0)
        tensor = tensor.expand(n_head, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        return alibi_mask.permute(1, 0, 2).unsqueeze(2)

    # 固定基频: rope_theta
    # 自定义基频: rope_given_inv_feq_str
    # 分段基频: rope_theta/rope_given_inv_feq_str + rope_vanilla_theta + rope_keep_local_base_windows
    def _get_cos_sin_table(self, max_seq_len, dtype, device, params):
        given_inv_feq_str = params.rope_given_inv_feq_str
        if given_inv_feq_str:
            inv_freq = torch.FloatTensor([float(invf) for invf in given_inv_feq_str.split(',')], device=device)
            if len(inv_freq) != params.dim // 2:
                logger.error("error: only support len(inv_freq) == dim/2 ,check your inv_freq length", 
                             ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise AssertionError('given_inv_feq_str: length not match head_dim/2')
        else:
            inv_freq = 1.0 / (params.rope_theta ** (torch.arange(0,
            params.dim, 2, device=device).float() / params.dim))

        seq = torch.arange(max_seq_len, device=device).float() + params.offset
        freqs = torch.outer(seq, inv_freq)

        if params.rope_keep_local_base_windows:
            keep_local_base_windows = [int(w) for w in params.rope_keep_local_base_windows.split(',')]
            if len(keep_local_base_windows) != params.dim // 2:
                logger.error(
                    "error: only support len(keep_local_base_windows) == dim/2 ,check your base_windows length", 
                    ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise AssertionError('keep_local_base_windows: length not match head_dim/2')

            inv_freq_base = 1.0 / (params.rope_vanilla_theta ** (torch.arange(0,
            params.dim, 2, device=device).float() / params.dim))
            freqs_base = torch.outer(seq, inv_freq_base)
            freqs_after_window = freqs + torch.tensor(keep_local_base_windows) * (inv_freq_base - inv_freq)
            for idx, i_keep_local_base_window in enumerate(keep_local_base_windows):
                freqs[:, idx] = torch.cat((
                    freqs_base[:i_keep_local_base_window, idx],
                    freqs_after_window[i_keep_local_base_window:, idx]
                ))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation（ks）
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * params.rope_mscale).to(dtype).to(device), \
        (emb.sin() * params.rope_mscale).to(dtype).to(device)

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        if self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                             device,
                                                             max_seq_len)

        else:
            scaling_type = self.config.rope_scaling.rope_type
            if scaling_type is None:
                scaling_type = self.config.rope_scaling.type
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                                 device,
                                                                 max_seq_len)
            elif scaling_type == "llama3":
                self.rotary_embedding.update_llama3_cos_sin_cache_total(self.config,
                                                                        dtype,
                                                                        device,
                                                                        max_seq_len)
            elif scaling_type == "dynamic":
                if self.warmup:
                    print_log(self.tp_rank, logger.info,
                              f"Using dynamic ntk feature to support long seq, please export LONG_SEQ_ENABLE=1, "
                              f"now LONG_SEQ_ENABLE is {int(ENV.long_seq_enable)}.")
            else:
                logger.error("error: only support scaling type: linear, dynamic, check your config.json: scaling type", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                raise ValueError("Unknown RoPE scaling type, check your config.json: rope_scaling type")

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()


