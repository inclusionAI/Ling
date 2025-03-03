# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from abc import abstractmethod
from typing import Optional, List, Tuple

import torch
import torch_npu
from transformers.configuration_utils import PretrainedConfig

from .model_utils import BaseModel
from ...models import InferenceMode
from ...utils.env import ENV
from ...utils.log import logger, print_log
from ...utils.initial import load_atb_speed, NPUSocInfo, is_lcoc_enable
from ...utils.layers import PositionRotaryEmbedding, AttentionMask
from ...utils.op_backend import OpBackend
from ...utils.adapter_manager import AdapterIdsType
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.weights import Weights


class FlashForCausalLM(BaseModel):
    """
    Base class for causal language model using paged attention, built with Python graph.

    Args:
        config (PretrainedConfig): The configuration for the model.
        weights (Weights): The weights for the model.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, config: PretrainedConfig, weights: Weights, **kwargs):
        super().__init__()
        load_atb_speed()
        self.model = None
        self.lm_head = None
        self.config = config
        self.soc_info = NPUSocInfo()

        self.inference_mode = kwargs.get("inference_mode")

        self.num_attention_heads = config.num_attention_heads
        if hasattr(config, 'num_key_value_heads'):
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads

        if hasattr(config, 'rope_theta'):
            self.rope_theta = config.rope_theta
        else:
            self.rope_theta = 10000.0
        if hasattr(config, 'rope_scaling') and self.config.rope_scaling is not None:
            self.scaling_factor = self.config.rope_scaling.factor
        else:
            self.scaling_factor = 1.0
        self.hidden_size = config.hidden_size
        self.head_size = config.head_dim \
                         if config.model_type == "gemma" \
                         else self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.device = weights.device
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        print_log(self.tp_rank, logger.info, self.soc_info)
        self.lcoc_enable = is_lcoc_enable(self.soc_info.need_nz)
        self.speculate_enable = self.inference_mode == InferenceMode.SPECULATE \
                                or self.inference_mode == InferenceMode.PREFIXCACHE  # 使用prefix cache特性也需要使用并行解码算子
        self.prefix_cache_enable = self.inference_mode == InferenceMode.PREFIXCACHE
        self.compress_head_enable = ENV.compress_head_enable
        self.split_fuse_enable = self.inference_mode == InferenceMode.SPLITFUSE
        if self.config.quantization_config.reduce_quant_type is not None:
            if self.tp_world_size <= 1:
                self.config.quantization_config.reduce_quant_type = None
            else:
                self.lcoc_enable = False
        # if num_key_value_heads is nondivisible 
        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
            // self.tp_world_size

        self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                               device="cpu", scaling_factor=self.scaling_factor) \
            .to(self.device)
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize = config.quantize
        self.dtype = weights.dtype

        self.max_base_len = 128
        self.attn_mask = AttentionMask.static(self.max_base_len, dtype=self.dtype)

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")

        # for ascend init
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.ascend_kcache_shape = None
        self.ascend_vcache_shape = None

        self.acl_encoder_operation_inputs: list[None | torch.Tensor] \
            = [None] * self.get_in_tensor_size(encoder=True)
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] \
            = [None] * self.get_in_tensor_size(encoder=False)

        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int).to(self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).to(self.device)
        self.attn_mask_fake = self.attn_mask \
            .get_attn_mask(1, dtype=self.dtype, device="cpu") \
            .to(self.device)

        self.acl_param = None
        self.cos_embed = None
        self.sin_embed = None

        self.adapter_manager = None
        self.num_lora_weight_per_layer = 14
        self.adapter_ids = None
        
        self.attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        self.mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        self.attn_decode_backend = OpBackend.ATB
        if self.attn_decode_backend == OpBackend.ACLNN:
            print_log(self.tp_rank, logger.warning,
                      "If the model's max_positional_embedding is large, "
                      "AclNN attention backend may result in NPU out of memory.")

    @staticmethod
    def update_adapter_weights(adapter_weights: list[torch.Tensor], in_tensor: list[torch.Tensor], start_idx:int):
        """Update adapter weights."""
        for i, weight in enumerate(adapter_weights):
            # 这里+1是需要跳过seq_len_cum_sum
            in_tensor[start_idx + 1 + i] = weight

    def update_adapter_manager(self):
        """Update adapter manager."""
        self.adapter_manager.base_model = self
        # +1 是因为Lora旁路需要多一个seq_len_cum_sum入参
        self.acl_encoder_operation_inputs.extend([None] * (self.num_lora_weight_per_layer * self.num_layers + 1))
        self.acl_decoder_operation_inputs.extend([None] * (self.num_lora_weight_per_layer * self.num_layers + 1))

    def get_in_tensor_size(self, encoder: bool = True) -> int:
        """Get input tensor size."""
        return 9

    def weight_format_cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast weight to nz format if based on SOC info."""
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        print_log(self.tp_rank, logger.info, f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor
    
    def process_adapter_ids(self, adapter_ids: None | List[str | None]) -> List[str]:
        """Preprocess adapter ids."""
        if self.adapter_manager is None:
            return []
        effective_adapter_ids = self.adapter_manager.preprocess_adatper_ids(adapter_ids)
        return effective_adapter_ids

    def prepare_adapter_weights(self, adapter_ids: None | List[str | None]) -> List[torch.Tensor]:
        """Prepare adapter weights."""
        need_update = self.adapter_manager.update_adapter(adapter_ids)
        # 不需要更新adapter weights的场景
        if not need_update:
            return []
        # 更新adapter weights
        return self.adapter_manager.get_adapters(adapter_ids)

    def calculate_adapter_group_size(
            self, adapter_ids: None | List[str | None],
            input_lengths: torch.Tensor, is_prefill: bool = False
    ) -> torch.Tensor:
        """Calculate the adapter group size."""
        if len(adapter_ids) == 1:
            return self.placeholder
        elif self.adapter_manager.previous_adapter_ids.record_type == AdapterIdsType.MIXED:
            if is_prefill:
                cum_group_size = torch.cumsum(input_lengths, dim=0, dtype=torch.int64)
            else:
                cum_group_size = torch.arange(1, input_lengths.shape[0] + 1, dtype=torch.int64, device=self.device)
        else:
            active_adapters_count = len(self.adapter_manager.adapter_info_registry) - 1  # exclude *sort
            adapter_indexes = []
            for adapter_id in adapter_ids:
                adapter_indexes.append(self.adapter_manager.adapter_info_registry.get(adapter_id).idx)
            labels = torch.tensor(adapter_indexes, device=self.device, dtype=torch.int64)
            unique_labels = torch.arange(0, active_adapters_count, dtype=torch.int64, device=self.device)
            group = torch.zeros_like(unique_labels).scatter_add_(0, labels, input_lengths.to(torch.int64))
            cum_group_size = torch.cumsum(group, dim=0, dtype=torch.int64)
        return cum_group_size

    @abstractmethod
    def init_ascend_operations(self, config: PretrainedConfig):
        """Abstract method to initialize Ascend operations."""
        pass

    @abstractmethod
    def init_ascend_weight(self):
        """Abstract method to initialize Ascend weights."""
        pass
    
    def get_weight_wrapper(self) -> WeightWrapper:
        """Get weight and regist embedding, layer, quant (if needed), norm and lmhead."""
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, self.attn_wrapper, self.mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.reduce_quant_type is not None:
                weight_wrapper.register_layer_reducequant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def get_coder_param(self) -> Tuple[dict, dict]:
        """Set coder param and get encoder/decoder params."""
        weight_wrapper = self.get_weight_wrapper()
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
            "enableAddNorm": False,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_dim,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False,
            "kvQuant": self.config.quantization_config.kv_quant_type is not None,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or rank_table_file else "lccl",
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": position_embedding_type,
            "isUnpadInputs": True,
        }
        
        encoder_param = {
            **coder_param, "isPrefill": True,
            "supportLcoc": self.lcoc_enable,
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "supportLcoc": False
        }
        return encoder_param, decoder_param

    def get_adapter_ids(self, **kwargs):
        """Get adapter ids from keywords."""
        if self.adapter_manager is not None:
            self.adapter_ids = kwargs.get("adapter_ids")

    def init_position_rotary_embedding(self, position_ids: torch.Tensor, max_seq_len: int):
        """Initialze rope."""
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        if self.num_attention_heads == self.num_key_value_heads:
            self.cos_embed, self.sin_embed = self.rotary_embedding.get_cos_sin_cached_total(position_ids)
        else:
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_kvcache(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Initialzie key-value cache."""
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
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs) -> Tuple[list, str]:
        """
        Prepare the inputs for Ascend acl operation graph.

        Args:
            input_ids (torch.Tensor): The input tensor.
            position_ids (torch.Tensor): The position ids tensor.
            is_prefill (bool): Whether the inference mode is prefill.
            kv_cache (List[Tuple[torch.Tensor, torch.Tensor]]): Key-value cache.
            block_tables (torch.Tensor): Input block tables.
            slots (torch.Tensor): Input slots.
            input_lengths (torch.Tensor): Input lengths.
            max_seq_len (torch): Maximum sequence length.
            lm_head_indices (torch.Tensor, optional): LM head indices. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            list: A list of Ascend acl encoder operation inputs.
            str: A json formatted string contains operation parameters.
        """
        self.init_position_rotary_embedding(position_ids, max_seq_len)
        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = atten_mask.view(1, pad_maxs, pad_maxs // 16, 16).transpose(1, 2)
                torch_npu.npu_format_cast_(atten_mask, 29)
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                          kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = self.cos_embed
            self.acl_encoder_operation_inputs[3] = self.sin_embed
            if self.dtype == torch.bfloat16:
                self.acl_encoder_operation_inputs[4] = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            else:
                self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[8] = lm_head_indices.to(torch.int64)
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.cos_embed
            self.acl_decoder_operation_inputs[3] = self.sin_embed
            if self.dtype == torch.bfloat16:
                self.acl_decoder_operation_inputs[4] = torch.zeros(input_lengths.size(0),
                                                                   self.num_attention_heads,
                                                                   1, input_lengths.max(),
                                                                   dtype=self.dtype,
                                                                   device=input_ids.device)
            else:
                self.acl_decoder_operation_inputs[4] = self.attn_mask_fake
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs: list,
                                acl_param: str,
                                is_prefill: bool):
        """Execute the Ascend acl operator."""
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        return acl_hidden_state

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
        """
        Forward pass of the model.'

        Args:
            input_ids (torch.Tensor): The input ids tensor.
            position_ids (torch.Tensor): The position ids tensor.
            is_prefill (bool): Whether the inference mode is prefill.
            kv_cache (List[Tuple[torch.Tensor, torch.Tensor]]): Key-value cache.
            block_tables (torch.Tensor): Input block tables.
            slots (torch.Tensor): Input slots.
            input_lengths (torch.Tensor): Input lengths.
            max_seq_len (torch): Maximum sequence length.
            lm_head_indices (torch.Tensor, optional): LM head indices. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            torch.Tensor: Output logits.
        """
        if not self.ascend_weight:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits
