# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
import importlib
from typing import Optional, List, Tuple

import torch
import torch_npu

from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.models.qwen.base_qwen import QwenRotaryEmbedding

from .modeling_qwen import FlashQwenModel
from .config_qwen import QwenConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType


FLASH_CAUSAL_DEVICE_NAME = "npu"
CPP_QWEN_MODEL_CLASS_NAME = "qwen_QwenDecoderModel"


class FlashQwenForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.acl_decoder_regression_operation = None
        super().__init__(config, weights, **kwargs)
        self.transformer = FlashQwenModel(config, weights)
        if config.tie_word_embeddings:
            self.lm_head = load_column_multi(
                config,
                prefixes=["model.embed_tokens"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        else:
            self.lm_head = load_column_multi(
                    config,
                    prefixes=["lm_head"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )

        self.config = config
        if self.dtype != torch.float16:
            msg = f"unsupported type: {self.dtype}, just support `float64`, please modify the `torch_dtype`"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        self.attn_mask_fake = self.attn_mask.get_attn_mask(1, dtype=torch.float16, device=FLASH_CAUSAL_DEVICE_NAME)
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.acl_param = None
        self.acl_operation_inputs = None
        self.ascend_weight = None

        self.long_seq_enable = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn
        if self.use_logn_attn:
            logn_list = [
                math.log(i, config.seq_length) if i > config.seq_length else 1
                # 防止32K长序列越界
                for i in range(1, 65536)
            ]
            if self.soc_info.need_nz:
                self.logn_tensor = torch.tensor(logn_list, dtype=torch.float16,
                                                device=FLASH_CAUSAL_DEVICE_NAME)[:, None, None]
            else:
                self.logn_tensor = torch.tensor(logn_list, dtype=torch.float32,
                                                device=FLASH_CAUSAL_DEVICE_NAME)[:, None, None]
            if config.rotary_pct == 1.0:
                self.rotary_ndims = None
            elif config.rotary_pct < 1:
                self.rotary_ndims = int(config.kv_channels * config.rotary_pct)
            else:
                msg = "rotary_pct > 1"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise RuntimeError(msg)
            self.rotary_emb = QwenRotaryEmbedding(
                self.rotary_ndims if self.rotary_ndims else config.kv_channels,
                base=config.rotary_emb_base
            )
        
        self.regression_acl_operation_num = -1

    def init_ascend_operations(self, config: QwenConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        if self.prefix_cache_enable:
            self.acl_decoder_regression_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        logger.info(">>>> qwen_QwenDecoderModel is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1', wrapper_name='attn',
            pack_name='c_attn', sep_names=None,
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2', wrapper_name='mlp',
            pack_name='w2_w1', sep_names=None,
            down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.ln_2
                del layer.attn
                del layer.mlp
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
            "isFA": False,
            "isBF16": False,
            "withEmbedding": True,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "normEps": self.config.layer_norm_epsilon,
            "normType": NormType.RMS_NORM,
            "isUnpadInputs": True,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "enableLogN": self.use_logn_attn,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "quantGroupSize": self.config.quantization_config.group_size,
            "kvQuant": False,
            "isLongSeq": self.long_seq_enable,
            "rankTableFile": ENV.rank_table_file,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers,
            "enableQScale": self.config.num_hidden_layers == 32 or \
                            self.config.num_hidden_layers == 80
        }
        acl_param_encoder = json.dumps(
            {
                **acl_param_dict, "isPrefill": True, 
                "enableLcoc": self.lcoc_enable
            }
        )
        acl_param_decoder = json.dumps(
            {
                **acl_param_dict, "isPrefill": False, "enableLcoc": False,
                "enableSpeculate": self.speculate_enable,
                "enablePrefixCache": self.prefix_cache_enable
            }
        )
        
        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)
        
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

        if self.prefix_cache_enable:
            self.init_prefix_cache_regression_weight(acl_param_dict)
    
    def init_prefix_cache_regression_weight(self, coder_param):
        # prefix cache特性多加一张图用于自回归decode
        decoder_regression_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False,
            "enableSpeculate": False
        }
        self.acl_decoder_regression_operation.set_param(json.dumps({**decoder_regression_param}))
        self.acl_decoder_regression_operation.set_weight(self.ascend_weight)

    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / self.config.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor, is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor, input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        q_lens = 1
        spec_mask = None
        self.acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        if self.speculate_enable:
            q_lens = kwargs.get('q_lens', [])
            spec_mask = kwargs.get('spec_mask', None)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            })
            if not is_prefill:
                q_lens = torch.tensor(q_lens).to(self.device).to(torch.int32)
        cos_table, sin_table = None, None
        logn_tensor = None

        if self.long_seq_enable:
            if is_prefill:
                self.rotary_embedding.clear_ntk_cache(input_lengths.shape[0])
                seq_start = torch.zeros(input_lengths.shape, dtype=torch.int32, device=FLASH_CAUSAL_DEVICE_NAME)
                seq_end = input_lengths
            else:
                seq_start = input_lengths - q_lens
                seq_end = input_lengths
            for i in range(input_lengths.shape[0]):
                ntk_alpha = self.get_ntk_alpha(input_lengths[i])
                self.rotary_embedding.dynamic_ntk_inv_freq(self.config, input_lengths[i].item(), 
                                                           self.device, ntk_alpha, i)
            logn_tensor_list = [
                self.logn_tensor[seq_start[i]:seq_end[i], :, :].repeat(1, self.num_attention_heads, self.head_size)
                for i in range(input_lengths.shape[0])
            ]
            logn_tensor = torch.cat(logn_tensor_list, dim=0)

            offset = 0
            position_ids_offset = position_ids.clone()
            for i in range(input_lengths.shape[0]):
                if is_prefill:
                    position_ids_offset[seq_start[i] + offset:seq_end[i] + offset] \
                        += self.rotary_embedding.position_ids_offset[i]
                    offset += input_lengths[i]
                else:
                    position_ids_offset[i] += self.rotary_embedding.position_ids_offset[i]
            
            pos_lens = self.rotary_embedding.pos_lens
            inv_freqs = self.rotary_embedding.ntk_inv_freqs
            position_ids = self.rotary_embedding.position_ids_expanded
        
        else:

            if self.config.num_hidden_layers == 40:
                self.rotary_embedding.update_cos_sin_cache_total(
                    self.dtype,
                    self.device,
                    self.max_position_embeddings
                )
            else:
                self.rotary_embedding.update_cos_sin_cache_total(
                    torch.float32,
                    self.device,
                    self.max_position_embeddings
                ) 
            cos_table = self.rotary_embedding.get_cos_cached_total()
            sin_table = self.rotary_embedding.get_sin_cached_total()

        use_regression = False
        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            if self.soc_info.need_nz:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
        else:
            if self.prefix_cache_enable and len(q_lens) == 0:  # 开启prefix cache时q_lens为空时使用自回归
                use_regression = True
            attention_mask = self.attn_mask_fake

            if self.speculate_enable and not use_regression:
                # 单草稿不用传mask
                if spec_mask is None:
                    bs = len(q_lens)
                    sum_qlen = sum(q_lens)
                    atten_mask = self.attn_mask.get_attn_mask(max_seq_len, kv_cache[0][0].dtype,
                                                            kv_cache[0][0].device)
                    req_mask = torch.zeros(size=(sum_qlen, max_seq_len)).to(kv_cache[0][0].dtype) \
                                    .to(kv_cache[0][0].device)
                    start_row = 0
                    for i in range(bs):
                        start = input_lengths[i] - q_lens[i]
                        end = input_lengths[i]
                        end_row = start_row + q_lens[i]
                        req_mask[start_row:end_row] = atten_mask[start:end]
                        start_row += q_lens[i]
                else:
                    req_mask = spec_mask
                if self.soc_info.need_nz:
                    req_mask = self.transdata_operation.execute([req_mask])[0]
                attention_mask = req_mask
        
        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        if use_regression or not (is_prefill or self.prefix_cache_enable):
            lm_head_indices = self.lm_head_indices_fake

        self.acl_operation_inputs = [
            input_ids,  # IN_TENSOR_INPUTIDS
            position_ids, # IN_TENSOR_POSITIONIDS
            self.place_holder if self.long_seq_enable else cos_table,  # IN_TENSOR_COSEMBED
            self.place_holder if self.long_seq_enable else sin_table,  # IN_TENSOR_SINEMBED
            attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            logn_tensor if self.long_seq_enable else self.place_holder,  # IN_TENSOR_KV_CACHE_IDX
            self.place_holder,  # IN_TENSOR_TOKEN_OFFSET
            self.place_holder,  # IN_HOLDER
            input_lengths.to(torch.int32),  # IN_TENSOR_SEQ_LENGTHS
            lm_head_indices  # IN_TENSOR_LOGTIS_INDICES
        ]
        if self.long_seq_enable:
            self.acl_operation_inputs.append(inv_freqs)
            self.acl_operation_inputs.append(pos_lens)
            self.acl_operation_inputs.append(position_ids_offset)
        self.regression_acl_operation_num = len(self.acl_operation_inputs)

        if self.speculate_enable and not is_prefill and not use_regression:
            self.acl_operation_inputs.append(q_lens)  # qLen;

        return self.acl_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            # prefix cache自回归decode
            if self.prefix_cache_enable and len(acl_inputs) == self.regression_acl_operation_num:
                acl_model_out = self.acl_decoder_regression_operation.execute(acl_inputs, acl_param)
            else:
                acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            msg = "RuntimeError: enable the log to locate the fault."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError(msg) from e
        return acl_hidden_state

    def init_kvcache(self, kv_cache):
        kcache_shape_diff = self.ascend_kcache_shape != kv_cache[0][0].shape
        vcache_shape_diff = self.ascend_vcache_shape != kv_cache[0][1].shape
        kcache_id_diff = self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id_diff = self.ascend_vcache_id != id(kv_cache[0][1])
        kcache_diff = not self.ascend_kcache_id or kcache_id_diff or kcache_shape_diff
        vcache_diff = not self.ascend_vcache_id or vcache_id_diff or vcache_shape_diff
        if kcache_diff or vcache_diff:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            print_log(self.tp_rank, logger.info, f"<<<<<<< ori {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            if self.prefix_cache_enable:
                self.acl_decoder_regression_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")
