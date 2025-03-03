# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional, List, Tuple

import math
import torch
import torch_npu

from .modeling_qwen2 import FlashQwenModel
from .config_qwen2 import Qwen2Config
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.env import ENV
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers import load_column_multi, TensorHead, TensorEmbedding
from ...utils.log import logger, print_log
from ...utils.layers.norm.fast_layer_norm import NormType

CPP_QWEN_MODEL_CLASS_NAME = "qwen_QwenDecoderModel"


class FlashQwen2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.acl_decoder_regression_operation = None
        super().__init__(config, weights, **kwargs)

        model_prefix = kwargs.get("model_prefix", "model")
        lmhead_prefix = kwargs.get("lmhead_prefix", "lm_head")
        transformer_wte_parallel = kwargs.get("transformer_wte_parallel", True)
        self.skip_word_embedding = kwargs.get("skip_word_embedding", False)

        self.transformer = FlashQwenModel(config, weights, model_prefix=model_prefix, lmhead_prefix=lmhead_prefix)
        if not transformer_wte_parallel:
            self.transformer.wte = TensorEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )
            for p in self.transformer.wte.parameters():
                p.requires_grad = False

        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=False,
            )
        else:
            if config.tie_word_embeddings:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[f"{model_prefix}.embed_tokens"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
            else:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[f"{lmhead_prefix}"],
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

        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.acl_param = None
        self.ascend_weight = None
        self.acl_multi_lora_encoder_operation = None
        self.acl_multi_lora_decoder_operation = None

        self.long_seq_enable = False
        if hasattr(self.config, 'rope_scaling') and hasattr(self.config.rope_scaling, 'type') and \
                self.config.rope_scaling.type == 'yarn':
            self.long_seq_enable = True
            if self.config.rope_scaling.attention_factor is None:
                self.attention_factor = 1.0
            else:
                self.attention_factor = float(self.config.rope_scaling.attention_factor)
            if self.config.rope_scaling.factor is None:
                raise ValueError('config.rope_scaling.factor must be set in config.json')
            if self.config.rope_scaling.factor <= 1:
                self.mscale = self.attention_factor
            else:
                self.mscale = float((0.1 * math.log(self.config.rope_scaling.factor) + 1.0) * self.attention_factor)
        self.acl_operation_inputs = [None] * self.get_in_tensor_size(encoder=True, long_seq_enable=self.long_seq_enable)

    def update_adapter_manager(self):
        self.adapter_manager.base_model = self
        self.acl_operation_inputs.extend([None] * (self.num_lora_weight_per_layer * self.num_layers + 1))

    def get_in_tensor_size(self, encoder=True, long_seq_enable=False, regression=False):
        base_size = 12
        if long_seq_enable:
            base_size += 3
        if regression:
            return base_size
        if self.compress_head_enable:
            base_size += 2  # batch_wins, input_length_js
        if not encoder and self.speculate_enable:
            base_size += 1  # 1: q_len
        return base_size

    def init_ascend_operations(self, config: Qwen2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        if self.prefix_cache_enable:
            self.acl_decoder_regression_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        logger.info(">>>> qwen_QwenDecoderModel is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=None,
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=None,
            down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz and self.adapter_manager is None:
                del layer.attn
                del layer.ln_2
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
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
            "isBF16": self.dtype == torch.bfloat16,
            "skipWordEmbedding": self.skip_word_embedding,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "isUnpadInputs": True,
            "enableFA3": self.config.quantization_config.fa_quant_type is not None,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "quantGroupSize": self.config.quantization_config.group_size,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "enableLora": self.adapter_manager is not None,
            "isLongSeq": self.long_seq_enable,
            "enableAddNorm": False,
            "isYarn": self.long_seq_enable,
            "mscale": self.mscale if self.long_seq_enable else 1.0,
            "rankTableFile": ENV.rank_table_file,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers,
            "enableQScale": (self.config.transformers_version == "4.44.0" or
                            self.config.transformers_version == "4.43.1") and \
                            self.config.num_hidden_layers == 28 and \
                            self.soc_info.need_nz
                            # QwenCode2.5-7B-Instruct/Qwen2.5-7B/1.5B-Instruct模型时为True, 其他模型为False
        }
        encoder_param = {
            **acl_param_dict,
            "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "enableSplitFuse": self.split_fuse_enable
        }
        decoder_param = {
            **acl_param_dict,
            "isPrefill": False,
            "enableLcoc": False,
            "enableSpeculate": self.speculate_enable,
            "enablePrefixCache": self.prefix_cache_enable
        }

        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

        if self.adapter_manager is not None:
            self.acl_multi_lora_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_QwenDecoderModel")
            self.acl_multi_lora_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_QwenDecoderModel")

            encoder_param.update({"loraEnableGMM": True})
            self.acl_multi_lora_encoder_operation.set_param(json.dumps({**encoder_param}))
            decoder_param.update({"loraEnableGMM": True})
            self.acl_multi_lora_decoder_operation.set_param(json.dumps({**decoder_param}))

            self.acl_multi_lora_encoder_operation.set_weight(self.ascend_weight)
            self.acl_multi_lora_decoder_operation.set_weight(self.ascend_weight)

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
        if self.adapter_manager is not None:
            # 更新adapter
            adapter_ids = kwargs.get("adapter_ids")
            effective_adapter_ids = self.process_adapter_ids(adapter_ids)
            adapter_weights = self.prepare_adapter_weights(effective_adapter_ids)
            in_tensor_size = self.get_in_tensor_size(encoder=is_prefill)
            self.update_adapter_weights(adapter_weights, self.acl_operation_inputs, in_tensor_size)
        seqlen = "seqLen"
        self.acl_param = json.dumps({
            seqlen: input_lengths.tolist()
        })
        cos_table, sin_table = None, None
        if self.long_seq_enable:
            self.rotary_embedding.yarn_scaling_rotary_embedding(self.config, self.device, max_seq_len)
        else:
            self.rotary_embedding.update_cos_sin_cache_total(
                self.dtype,
                self.device,
                self.max_position_embeddings
            )
            cos_table = self.rotary_embedding.get_cos_cached_total()
            sin_table = self.rotary_embedding.get_sin_cached_total()

        q_lens = kwargs.get('q_lens', [])

        inv_freq = self.rotary_embedding.ntk_inv_freqs
        position_ids_expanded = self.rotary_embedding.position_ids_expanded
        pos_lens = self.rotary_embedding.pos_lens

        use_regression = False
        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(max_seq_len if self.split_fuse_enable else self.max_base_len,
                                                          self.dtype, self.device)
            # BF16 PA算子需要使用-10000.0表示mask掉的部分
            if self.split_fuse_enable and self.dtype == torch.bfloat16:
                attention_mask = attention_mask * -10000.0
            if self.soc_info.need_nz:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            if self.split_fuse_enable:
                self.acl_param = json.dumps({
                    seqlen: input_lengths.tolist(),
                    "qLen": q_lens
                })
                q_lens = torch.tensor(q_lens).to(self.device).to(torch.int32)
        else:
            if self.prefix_cache_enable and q_lens == []:  # 开启prefix cache时q_lens为空时使用自回归
                use_regression = True
            if self.skip_word_embedding:
                input_ids = self.transformer.wte(input_ids)
            attention_mask = self.attn_mask_fake

            if self.speculate_enable and not use_regression:
                spec_mask = kwargs.get('spec_mask', None)
                self.acl_param = json.dumps({
                    seqlen: input_lengths.tolist(),
                    "qLen": q_lens
                })
                q_lens = torch.tensor(q_lens).to(self.device).to(torch.int32)
                req_mask = spec_mask
                if self.soc_info.need_nz:
                    req_mask = self.transdata_operation.execute([req_mask])[0]
                attention_mask = req_mask

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        if use_regression or not (is_prefill or self.prefix_cache_enable):
            lm_head_indices = self.lm_head_indices_fake

        acl_operation_inputs_ = [
            input_ids,  # IN_TENSOR_INPUTIDS
            position_ids if not self.long_seq_enable else position_ids_expanded, # IN_TENSOR_POSITIONIDS
            self.place_holder if self.long_seq_enable else cos_table,  # IN_TENSOR_COSEMBED
            self.place_holder if self.long_seq_enable else sin_table,  # IN_TENSOR_SINEMBED
            attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            self.place_holder,  # IN_TENSOR_KV_CACHE_IDX
            self.place_holder,  # IN_TENSOR_TOKEN_OFFSET
            self.place_holder,
            input_lengths.to(torch.int32),  # IN_TENSOR_SEQ_LENGTHS
            lm_head_indices,  # IN_TENSOR_LOGTIS_INDICES
        ]  # 0-11
        if self.long_seq_enable:  # 12-14 for long_seq
            acl_operation_inputs_.append(inv_freq)
            acl_operation_inputs_.append(pos_lens)
            acl_operation_inputs_.append(position_ids)

        if self.speculate_enable and not is_prefill and not use_regression:
            acl_operation_inputs_.append(q_lens) # 15
        if is_prefill:
            if self.split_fuse_enable:
                acl_operation_inputs_.append(q_lens)
            if self.adapter_manager is not None:
                acl_operation_input_lora = self.calculate_adapter_group_size(
                    effective_adapter_ids, input_lengths.to(torch.int32), is_prefill=True)
                acl_operation_inputs_.append(acl_operation_input_lora) # 12
        else:
            if self.adapter_manager is not None:
                acl_operation_input_lora = self.calculate_adapter_group_size(
                    effective_adapter_ids, torch.ones_like(input_ids, device=self.device, dtype=torch.int64),
                    is_prefill=False)
                acl_operation_inputs_.append(acl_operation_input_lora) # 12

        if self.speculate_enable or self.split_fuse_enable:
            self.acl_operation_inputs = acl_operation_inputs_
        else:
            self.acl_operation_inputs[:len(acl_operation_inputs_)] = acl_operation_inputs_        
        return self.acl_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        has_multiple_adapter_ids = self.adapter_manager is not None and acl_inputs[12].shape[0] != 1

        if is_prefill:
            model_operation = self.acl_multi_lora_encoder_operation \
                if has_multiple_adapter_ids else self.acl_encoder_operation
        else:
            if self.prefix_cache_enable and len(acl_inputs) == \
                    self.get_in_tensor_size(encoder=False, long_seq_enable=self.long_seq_enable, regression=True):
                model_operation = self.acl_decoder_regression_operation  # prefix cache自回归decode
            elif has_multiple_adapter_ids:
                model_operation = self.acl_multi_lora_decoder_operation
            else:
                model_operation = self.acl_decoder_operation

        acl_model_out = model_operation.execute(acl_inputs, acl_param)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
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
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")