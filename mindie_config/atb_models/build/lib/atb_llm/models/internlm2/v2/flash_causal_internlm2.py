# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from typing import Optional, List, Tuple

import torch

from atb_llm.utils.env import ENV
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi, PositionRotaryEmbedding, TensorHead
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger, print_log
from .modeling_internlm2 import FlashInternlm2Model
from .config_internlm2 import Internlm2Config


class FlashInternlm2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, lmhead_prefix="output", model_prefix="model", **kwargs):
        self.cos_embed = None
        self.sin_embed = None
        self.ascend_weight = None
        self.acl_operation_inputs = None
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights, **kwargs)
        self.skip_word_embedding = False
        self.model = FlashInternlm2Model(config, weights, model_prefix)
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=lmhead_prefix,
                weights=weights,
                is_norm=False,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=[lmhead_prefix],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        self.rank = ENV.rank
        self.dim = self.head_size
        self.base = config.rope_theta
        self.scaling_factor = 1.0
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_scaling = config.rope_scaling
        self.max_seq_len_cached = self.max_position_embeddings
        self.rotary_embedding_device = 'cpu'
        if self.rope_scaling is None:
            print_log(self.rank, logger.info, 'now \033[33m scaling_type: base rope \033[0m')
            self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                                device=self.rotary_embedding_device, 
                                                                scaling_factor=self.scaling_factor).to(self.device)
        else:
            self.scaling_type = self.rope_scaling.type
            if self.scaling_type == "linear":
                print_log(self.rank, logger.info, f'now \033[33m scaling_type: {self.scaling_type} \033[0m')
                self.scaling_factor = self.rope_scaling.factor # t=t/scaling_factor
                self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                                device=self.rotary_embedding_device, 
                                                                scaling_factor=self.scaling_factor).to(self.device)
            elif self.scaling_type == "dynamic":
                print_log(self.rank, logger.info, f'now \033[33m scaling_type: {self.scaling_type} \033[0m')
                self.rope_scaling_factor = self.rope_scaling.factor # Dynamic NTK 外推方法的系数
                self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                                device=self.rotary_embedding_device, 
                                                                scaling_factor=self.scaling_factor,
                ).to(self.device)
            else:
                print_log(self.rank, logger.info, f'now \033[33m scaling_type: {self.scaling_type} \033[0m')
                logger.error("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.",
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")

        if self.dtype != torch.float16:
            logger.error(f"unsupported type: {self.dtype}, 当前仅支持`float16`类型，"
                           "请修改权重文件config.json中的`torch_dtype`字段",
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(f"unsupported type: {self.dtype}, 当前仅支持`float16`类型，"
                               "请修改权重文件config.json中的`torch_dtype`字段")
        self.config = config
        self.head_dim = config.hidden_size / config.num_attention_heads
        self.in_tensor_length = 13
        self.acl_encoder_operation_inputs = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs = [None] * self.in_tensor_length

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.max_seq_len_cached = self.max_position_embeddings
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: Internlm2Config):
        logger.info("using internlm2_parallel_DecoderModel")
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("internlm2_20b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("internlm2_20b_DecoderModel")

    def get_embedding_layer(self):
        return self.model.tok_embeddings

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='attention_norm',
            wrapper_name='attention',
            pack_name='wqkv',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='wo'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ffn_norm',
            wrapper_name='feed_forward',
            pack_name='w1_w3',
            sep_names=['w1', 'w3'],
            down_name='w2'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.tok_embeddings)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz and self.adapter_manager is None:
                del layer.attention
                del layer.ffn_norm
                del layer.feed_forward
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        rank_table_file = ENV.rank_table_file
        # 设置模型参数
        coder_param = {
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.config.vocab_size > 100000,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "kvQuant": self.config.quantization_config.kv_quant_type is not None,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": "ROPE",
            "splitWithStride": True,
        }
        encoder_param = {
            **coder_param, "isPrefill": True, "supportLcoc": self.lcoc_enable,
            "skipWordEmbedding": self.skip_word_embedding,
            "supportLora": self.adapter_manager is not None,
            "useImMask": True
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "supportLcoc": False,
            "supportLora": False
        }
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        encoder_weight = self.ascend_weight
        decoder_weight = self.ascend_weight.copy()
        if self.adapter_manager is not None:
            effective_adapter_ids = self.process_adapter_ids(self.adapter_ids)
            adapter_weights = self.prepare_adapter_weights(effective_adapter_ids)
            encoder_weight.extend(adapter_weights)
        self.acl_encoder_operation.set_weight(encoder_weight)
        self.acl_decoder_operation.set_weight(decoder_weight)

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
        # add dynamic
        self.max_seq_len_cached = max(self.max_position_embeddings, max_seq_len)
        # warm_up 阶段会传入max_seq_len=max_input_length，导致 max_seq_len_cached 开始就达到最大
        if (self.rope_scaling is None):
            # RotaryEmbedding
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                             self.device,
                                                             self.max_position_embeddings)
        elif (self.scaling_type == "dynamic") and (self.max_seq_len_cached > self.max_position_embeddings):
            # DynamicNTKScalingRotaryEmbedding
            base = self.base * (
                (self.rope_scaling_factor * self.max_seq_len_cached / self.max_position_embeddings)
                - (self.rope_scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.rotary_embedding = self.rotary_embedding.static(dim=self.head_size, base=base,
                                                            device=self.rotary_embedding_device,
                                                            scaling_factor=self.scaling_factor,
            ).to(self.device)
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                                self.device,
                                                                self.max_seq_len_cached)
        else: # LinearScalingRotaryEmbedding
            # 如果 max_input_length > max_position_embeddings, 需要重置 base 和 rotary_embedding.inv_freq
            self.rotary_embedding = self.rotary_embedding.static(dim=self.head_size, base=self.base,
                                                            device=self.rotary_embedding_device,
                                                            scaling_factor=self.scaling_factor,
            ).to(self.device)
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                                self.device,
                                                                self.max_position_embeddings)

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        if is_prefill:
            if self.skip_word_embedding and len(input_ids.shape) < 2:
                input_ids = self.model.tok_embeddings(input_ids) # input_embeddings

            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
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
            input_tokens = self.placeholder if self.skip_word_embedding else input_ids
            input_embeddings = input_ids if self.skip_word_embedding else self.placeholder

            if self.dtype == torch.bfloat16:
                input_atten_mask = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            else:
                input_atten_mask = atten_mask
        else:
            input_tokens = input_ids
            input_embeddings = self.placeholder
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            if self.dtype == torch.bfloat16:
                input_atten_mask = torch.zeros(input_lengths.size(0),
                                         self.num_attention_heads,
                                         1, input_lengths.max().item(),
                                         dtype=self.dtype,
                                         device=self.device)
            else:
                input_atten_mask = self.attn_mask_fake

        im_mask = kwargs.get("im_mask", None)
        self.acl_operation_inputs = [
            input_tokens,
            input_embeddings,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            input_atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32),
            lm_head_indices if is_prefill else self.lm_head_indices_fake,
            im_mask if self.adapter_manager is not None else self.placeholder,
        ]

        for ind, item in enumerate(self.acl_operation_inputs):
            logger.debug(f"{ind} {item.device}")
        return self.acl_operation_inputs, self.acl_param
