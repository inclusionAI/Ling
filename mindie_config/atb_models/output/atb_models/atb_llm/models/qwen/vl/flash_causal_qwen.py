# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import json
import math
from typing import Optional, List, Tuple
import torch
import numpy as np

from atb_llm.models.qwen.config_qwen import QwenConfig
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.qwen.modeling_qwen import FlashQwenModel
from atb_llm.models.qwen.vl.visual import VisionTransformer
from atb_llm.utils.layers import TensorEmbedding, load_column_multi
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.shm_utils import get_data_from_shm

_CPP_QWEN_MODEL_CLASS_NAME = "qwen_QwenDecoderModel"


class FlashQwenForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.transformer = FlashQwenModel(config, weights)
        # Rewrite wte to turn off tensor parallel
        self.transformer.wte = TensorEmbedding(
            prefix="transformer.wte", weights=weights
        )
        for p in self.transformer.wte.parameters():
            p.requires_grad = False
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config  # for quantize
        self.attn_mask_fake = self.attn_mask.get_attn_mask(
            1, dtype=self.dtype, device="npu"
        )
        self.place_holder = torch.tensor([1], dtype=self.dtype, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch(
            "TransdataOperation"
        )
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        # visual
        self.visual = VisionTransformer(**config.visual).to(
            device=weights.device, dtype=weights.dtype
        )
        ks = [k for k in self.visual.state_dict().keys()]
        for k in ks:
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"transformer.visual.{k}"), requires_grad=False
            )
            k_list = k.split(".")
            target_module = self.visual
            for nxt_module in k_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, k_list[-1], saved_weight)
        self.image_start_id = config.visual["image_start_id"]

        self.acl_operation_inputs = None
        self.sin_table = None
        self.cos_table = None
        self.ascend_weight = None

    def init_position_rotary_embedding(
        self, position_ids: torch.Tensor, max_seq_len: int
    ):
        if self.num_attention_heads == self.num_key_value_heads:
            (
                self.cos_embed,
                self.sin_embed,
            ) = self.rotary_embedding.get_cos_sin_total(
                position_ids, max_seq_len, self.dtype
            )
        else:
            self.rotary_embedding.update_cos_sin_cache_total(
                self.dtype, position_ids.device, max_seq_len
            )
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: QwenConfig):
        if self.num_key_value_heads != self.num_attention_heads:
            msg = "Qwen-VL does not support GQA"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(
            _CPP_QWEN_MODEL_CLASS_NAME
        )
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(
            _CPP_QWEN_MODEL_CLASS_NAME
        )
        logger.info(f">>>> {_CPP_QWEN_MODEL_CLASS_NAME} is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name="ln_1",
            wrapper_name="attn",
            pack_name="c_attn",
            sep_names=None,
            o_name="c_proj",
        )
        mlp_wrapper = MlpWrapper(
            norm_name="ln_2",
            wrapper_name="mlp",
            pack_name="w2_w1",
            sep_names=None,
            down_name="c_proj",
        )
        weight_wrapper = WeightWrapper(
            self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper
        )
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.attn
                del layer.ln_2
                del layer.mlp
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return (
            weight_wrapper.weights,
            weight_wrapper.linear_type,
            weight_wrapper.pack_quant_type,
            weight_wrapper.linear_transpose_types,
        )

    def init_ascend_weight(self):
        (
            self.ascend_weight,
            linear_types,
            pack_quant_configs,
            linear_transpose_types,
        ) = self.get_weights()

        acl_param_dict = {
            "isFA": False,
            "isBF16": self.dtype is torch.bfloat16,
            "skipWordEmbedding": True,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "normEps": self.config.layer_norm_epsilon,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "quantGroupSize": self.config.quantization_config.group_size,
            "kvQuant": False,
            "isUnpadInputs": True,
            "linearHasBias": [[True, False, False, False]]
            * self.config.num_hidden_layers,
        }
        acl_param_encoder = json.dumps(
            {**acl_param_dict, "isPrefill": True, "enableLcoc": self.lcoc_enable}
        )
        acl_param_decoder = json.dumps(
            {**acl_param_dict, "isPrefill": False, "enableLcoc": False}
        )

        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def prepare_inputs_for_ascend(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        acl_param = json.dumps({"seqLen": input_lengths.tolist()})
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype, self.device, self.max_position_embeddings
        )

        self.cos_table = self.rotary_embedding.get_cos_cached_total()
        self.sin_table = self.rotary_embedding.get_sin_cached_total()
        if is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(
                    range(hidden_states.shape[0]),
                    dtype=torch.int64,
                    device=hidden_states.device,
                )

        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                attention_mask = self.attn_mask.get_attn_mask(
                    pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            else:
                attention_mask = self.attn_mask.get_attn_mask(
                    self.max_base_len, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
        else:
            attention_mask = self.attn_mask_fake

        self.acl_operation_inputs = [
            hidden_states,
            position_ids,
            self.cos_table,
            self.sin_table,
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.place_holder,
            self.place_holder,
            self.place_holder,
            input_lengths.to(torch.int32),
            (lm_head_indices if is_prefill else self.lm_head_indices_fake),
        ]

        return self.acl_operation_inputs, acl_param

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)

        hidden_states = self.transformer.wte(input_ids)

        if is_prefill:
            if torch.any(torch.eq(input_ids, self.image_start_id)):
                bos_pos = torch.where(torch.eq(input_ids, self.image_start_id))[0]
                eos_pos = torch.where(torch.eq(input_ids, self.image_start_id + 1))[0]
                image_num = bos_pos.shape[0]
                images = []
                pixel_array = []
                for i in range(image_num):
                    shm_value = input_ids[bos_pos[i] + 1]
                    shape_value = input_ids[bos_pos[i] + 2]
                    shared_array = get_data_from_shm(
                        shm_value, shape_value, np.float32, self.device
                    )
                    pixel_array.append(shared_array)

                if pixel_array:
                    pixel_array = torch.cat(pixel_array, dim=0)
                    images = self.visual(pixel_array)
                else:
                    images = self.visual.encode(images)
                for i in range(image_num):
                    hidden_states[bos_pos[i] + 1 : eos_pos[i]] = images[i]

        acl_inputs, acl_param = self.prepare_inputs_for_ascend(
            hidden_states,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            **kwargs,
        )

        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits
