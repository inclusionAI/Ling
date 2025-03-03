# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from abc import abstractmethod
from typing import Optional, List, Tuple
from collections import defaultdict, OrderedDict

import torch
import torch_npu
from transformers.modeling_utils import PretrainedConfig

import _libatb_torch as atb

from .model_utils import BaseModel
from ...models import InferenceMode
from ...utils.env import ENV
from ...utils.log import logger, print_log
from ...utils.initial import load_atb_speed, NPUSocInfo, is_lcoc_enable
from ...utils.layers import PositionRotaryEmbedding, AttentionMask
from ...utils.weights import Weights

PREFILL = "prefill"
DECODE = "decode"


class AtbGraph(atb._GraphOperation):
    """
    A class for managing the graph operations.

    Args:
        op_name (str, optional): The name of the operation, defaults to `model`.
    """
    def __init__(self, op_name: str = 'model'):
        super().__init__(op_name)
        self.operations = []


class FlashForCausalLMATB(BaseModel):
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
        # weights相关
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.device = weights.device
        self.dtype = weights.dtype
        self.inference_mode = kwargs.get("inference_mode")
        self.speculate_enable = self.inference_mode == InferenceMode.SPECULATE

        # 硬件相关
        self.soc_info = NPUSocInfo()
        print_log(self.tp_rank, logger.info, self.soc_info)

        # config相关
        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize = config.quantize
        self.kv_quant = config.quantization_config.kv_quant_type
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

        self.lcoc_enable = is_lcoc_enable(self.soc_info.need_nz)
        self.compress_head_enable = ENV.compress_head_enable

        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
                                            // self.tp_world_size

        # positional embedding相关
        self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                               device="cpu", scaling_factor=self.scaling_factor) \
            .to(self.device)
        self.cos_embed = None
        self.sin_embed = None

        # mask相关
        self.max_base_len = 128
        self.attn_mask = AttentionMask.static(self.max_base_len, dtype=self.dtype)
        self.attn_mask_fake = self.attn_mask \
            .get_attn_mask(1, dtype=self.dtype, device="cpu") \
            .to(self.device)

        # kv cache相关
        self.kcache_id = None
        self.vcache_id = None

        # 模型结构、输入、输出、参数、权重相关
        self.model = None
        self.lm_head = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()

        # 模型组图相关
        self.prefill_graph = None
        self.decode_graph = None

        # transdata operation
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

    @property
    @abstractmethod
    def name(self):
        """Abstract method to get model name."""
        pass

    def init_position_rotary_embedding(self, position_ids: torch.Tensor, max_seq_len: int):
        """Initialze the rotary embedding."""
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        if self.num_attention_heads == self.num_key_value_heads:
            self.cos_embed, self.sin_embed = self.rotary_embedding.get_cos_sin_cached_total(position_ids)
        else:
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_kvcache(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Initialize the key-value cache."""
        kcache_id = not self.kcache_id or self.kcache_id != id(kv_cache[0][0])
        vcache_id = not self.vcache_id or self.vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            k_caches, v_caches = map(list, zip(*kv_cache))
            print_log(self.tp_rank, logger.info, f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            for i, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
                k_cache_name = f"layer_{i}_k_cache"
                v_cache_name = f"layer_{i}_v_cache"
                self.weight.update({k_cache_name: k_cache, v_cache_name: v_cache})
            self.prefill_graph.set_weights(self.weight)
            self.decode_graph.set_weights(self.weight)
            self.kcache_id = id(kv_cache[0][0])
            self.vcache_id = id(kv_cache[0][1])
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.kcache_id} id of vcache is {self.vcache_id}")

    def get_weights(self) -> OrderedDict:
        """Get weights."""
        weights_dict = OrderedDict()
        weights_dict.update(self.model.get_weights(self.model_prefix))
        weights_dict.update(self.lm_head.linear.get_weights(self.lm_head_prefix))
        return weights_dict

    def init_graph(self):
        """Initialze weight, prefill graph and decode graph."""
        # 获取权重键值对
        self.weight = self.get_weights()
        # 创建atb graph
        self.prefill_graph = AtbGraph(f"{self.name}_prefill_graph")
        self.build_graph(self.prefill_graph, is_prefill=True)
        self.decode_graph = AtbGraph(f"{self.name}_decode_graph")
        self.build_graph(self.decode_graph, is_prefill=False)

    @abstractmethod
    def build_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """Abstract method to build computation graph."""
        pass

    @abstractmethod
    def get_in_tensor_names(self, is_prefill: bool):
        """Abstract method to get input tensor names."""
        pass

    @abstractmethod
    def get_out_tensor_names(self):
        """Abstract method to get output tensor names."""
        pass

    @abstractmethod
    def prepare_inputs(self, input_ids: torch.Tensor,
                       position_ids: torch.Tensor,
                       is_prefill: bool,
                       kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                       block_tables: torch.Tensor,
                       slots: torch.Tensor,
                       input_lengths: torch.Tensor,
                       max_seq_len: int,
                       lm_head_indices: Optional[torch.Tensor] = None,
                       **kwargs):
        """Abstract method to prepare inputs for inference."""
        pass

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
        Forward pass of the model: call prefill graph or decode graph to execute computation,
            according to the `is_prefill`.

        Args:
            input_ids (torch.Tensor): Input ids.
            position_ids (torch.Tensor): Position ids.
            is_prefill (bool): Whether the model is in prefill mode.
            kv_cache (List[Tuple[torch.Tensor, torch.Tensor]]): Key-value cache.
            block_tables (torch.Tensor): Input block tables.
            slots (torch.Tensor): Input slots.
            input_lengths (torch.Tensor): Input lengths.
            max_seq_len (int): Maximum sequence length.
            lm_head_indices (torch.Tensor, optional): LM head indices. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            torch.Tensor: Output logits.
        """
        self.init_kvcache(kv_cache)
        self.prepare_inputs(input_ids, position_ids, is_prefill, kv_cache,
                            block_tables, slots, input_lengths, max_seq_len,
                            lm_head_indices)
        if is_prefill:
            atb_model_out = self.prefill_graph.forward(self.graph_inputs[PREFILL], self.graph_outputs[PREFILL],
                                                       self.graph_param[PREFILL])
        else:
            atb_model_out = self.decode_graph.forward(self.graph_inputs[DECODE], self.graph_outputs[DECODE],
                                                      self.graph_param[DECODE])

        try:
            logits = atb_model_out[self.get_out_tensor_names()[0]]
        except IndexError as e:
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        return logits
