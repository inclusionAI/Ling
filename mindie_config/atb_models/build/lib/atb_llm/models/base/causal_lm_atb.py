# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from abc import abstractmethod
from typing import Optional, List, Tuple, Union
from collections import defaultdict, OrderedDict

import math
import torch
from torch import nn
import torch_npu
from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

import _libatb_torch as atb

from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.log import logger
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo, is_lcoc_enable
from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.weights import Weights

PREFILL = "prefill"
DECODE = "decode"


def _make_causal_mask(
        input_ids_or_embedding_shape: torch.Size,
        dtype: torch.dtype, device: torch.device,
        past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_or_embedding_shape[:2]
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class CausalLMATB(PreTrainedModel):
    """
    Base class for causal language models, built with Python graph.

    Args:
        config (PretrainedConfig): The configuration of the model.
        weights (Weights): The weights of the model.
        **kwargs (dict, optional): Additional keyword arguments.
    """
    def __init__(self, config: PretrainedConfig, weights: Weights, **kwargs):
        super().__init__(config)
        load_atb_speed()
        self.config = config
        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)

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
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.lcoc_enable = is_lcoc_enable(self.soc_info.need_nz)

        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = self.num_key_value_heads // self.tp_world_size

        self.batch_num = 0
        self.mask_full = None
        self.mask_inc = None

        self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                               device="cpu", scaling_factor=self.scaling_factor) \
            .to(weights.device)
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize = config.quantize
        self.kv_quant = config.quantization_config.kv_quant_type
        self.kv_dtype = self.dtype if self.kv_quant is None else torch.int8

        self.token_offset = None
        self.seq_len_encoder = None
        self.seq_len_decoder = None
        self.acl_param_seq_len_decoder = None
        self.k_cache = None
        self.v_cache = None
        self.past_key_values_length = 0
        self.nz_dim = 16

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
        """Abstract property for getting self.name."""
        pass
    
    def weight_format_cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Casts the weight tensor to nz format based on NPU requirement."""
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def get_weights(self) -> OrderedDict:
        """Get the weights of the model."""
        weights_dict = OrderedDict()
        weights_dict.update(self.model.get_weights(self.model_prefix))
        weights_dict.update(self.lm_head.linear.get_weights(self.lm_head_prefix))
        return weights_dict

    def init_graph(self):
        """Initialize the graph and weights."""
        # 获取权重键值对
        self.weight = self.get_weights()
        self.graph_inputs[PREFILL].update(**self.weight)
        self.graph_inputs[DECODE].update(**self.weight)
        # 创建atb graph
        self.prefill_graph = AtbGraph(f"{self.name}_prefill_graph")
        self.build_graph(self.prefill_graph, is_prefill=True)
        self.decode_graph = AtbGraph(f"{self.name}_decode_graph")
        self.build_graph(self.decode_graph, is_prefill=False)

    @abstractmethod
    def build_graph(self, graph: atb._GraphOperation, is_prefill: bool):
        """Abstract method for building graph."""
        pass

    @abstractmethod
    def get_in_tensor_names(self, is_prefill: bool):
        """Abstract method for getting input tensor names."""
        pass

    @abstractmethod
    def get_out_tensor_names(self):
        """Abstract method for getting output tensor names."""
        pass

    def init_kvcache(self, input_ids_or_embedding: Union[torch.LongTensor, torch.FloatTensor],
                     past_key_value: Optional[List[torch.FloatTensor]]):
        """
        Initialize the key-value cache.

        Args:
            input_ids_or_embedding (torch.LongTensor or torch.FloatTensor): The input embedding tokens or ids.
            past_key_value (List[torch.FloatTensor], optional): The past key-value cache.
        """
        batch_size = input_ids_or_embedding.shape[0]

        if batch_size != self.batch_num:
            self.batch_num = batch_size
            self.token_offset = torch.full(
                (self.batch_num,), 0, dtype=torch.int32, device=input_ids_or_embedding.device
            )
            self.seq_len_encoder = torch.full(
                (self.batch_num,), 1, dtype=torch.int32, device=input_ids_or_embedding.device
            )
            self.seq_len_decoder = torch.full(
                (self.batch_num,), 1, dtype=torch.int32, device=input_ids_or_embedding.device
            )
            self.acl_param_seq_len_decoder = [1] * self.batch_num
            self.mask_full = torch.zeros(
                (self.batch_num, self.max_position_embeddings, self.max_position_embeddings),
                dtype=self.dtype, device=input_ids_or_embedding.device
            )

            if not self.soc_info.need_nz:
                self.k_cache = [torch.zeros(self.batch_num,
                                        self.max_position_embeddings,
                                        self.num_key_value_heads * self.head_size, device=input_ids_or_embedding.device,
                                        dtype=self.dtype) for _ in range(self.num_layers)]
                self.v_cache = [torch.zeros(self.batch_num,
                                        self.max_position_embeddings,
                                        self.num_key_value_heads * self.head_size, device=input_ids_or_embedding.device,
                                        dtype=self.dtype) for _ in range(self.num_layers)]
            else:
                self.k_cache = [torch_npu.npu_format_cast_(torch.zeros(self.batch_num,
                                math.ceil(self.num_key_value_heads * self.head_size / self.nz_dim),
                                self.max_position_embeddings, self.nz_dim, device=input_ids_or_embedding.device,
                                dtype=self.dtype), 29) for _ in range(self.num_layers)]
                torch.npu.empty_cache()
                self.v_cache = [torch_npu.npu_format_cast_(torch.zeros(self.batch_num,
                                math.ceil(self.num_key_value_heads * self.head_size / self.nz_dim),
                                self.max_position_embeddings, self.nz_dim, device=input_ids_or_embedding.device,
                                dtype=self.dtype), 29) for _ in range(self.num_layers)]
                torch.npu.empty_cache()

        if past_key_value:
            self.k_cache = past_key_value[0]
            self.v_cache = past_key_value[1]
            self.past_key_values_length = self.token_offset[0]
            self.token_offset[:] = self.token_offset[0] + 1
        else:
            self.past_key_values_length = 0
            self.token_offset[:] = input_ids_or_embedding.shape[1]
            self.seq_len_encoder[:] = input_ids_or_embedding.shape[1]
        for i, (k_cache, v_cache) in enumerate(zip(self.k_cache, self.v_cache)):
            k_cache_name = f"layer_{i}_k_cache"
            v_cache_name = f"layer_{i}_v_cache"
            self.graph_inputs[PREFILL].update({k_cache_name: k_cache, v_cache_name: v_cache})
            self.graph_inputs[DECODE].update({k_cache_name: k_cache, v_cache_name: v_cache})

    def init_position_ids(self, input_ids_or_embedding: Union[torch.LongTensor, torch.FloatTensor],
                          position_ids: Optional[torch.LongTensor]) -> torch.Tensor:
        """
        Initialize the position ids.

        Args:
            input_ids_or_embedding (torch.LongTensor or torch.FloatTensor): The input embedding tensors or ids.
            position_ids (torch.LongTensor or None): The position ids tensor.
        
        Returns:
            torch.LongTensor: The position ids tensor.
        """
        seq_length = input_ids_or_embedding.shape[1]
        device = input_ids_or_embedding.device

        if position_ids is None:
            position_ids = torch.arange(
                self.past_key_values_length, seq_length + self.past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        return position_ids

    def init_mask(self, input_ids_or_embedding: Union[torch.LongTensor, torch.FloatTensor],
                  attention_mask: Optional[torch.Tensor]):
        """
        Initialize the causal mask for the model.

        Args:
            input_ids_or_embedding (torch.LongTensor or torch.FloatTensor): The input embedding tensors or ids.
            attention_mask (torch.Tensor, optional): The attention mask tensor.
        """
        batch_size, seq_length = input_ids_or_embedding.shape[0], input_ids_or_embedding.shape[1]
        device = input_ids_or_embedding.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_ids_or_embedding.shape,
                self.dtype,
                device=device,
                past_key_values_length=self.past_key_values_length,
            )
        attention_mask = _expand_mask(attention_mask, self.dtype, tgt_len=seq_length).to(device)
        attention_mask = attention_mask if combined_attention_mask is None else attention_mask + combined_attention_mask
        dim_0 = attention_mask.shape[2]
        dim_1 = attention_mask.shape[3]
        if not self.soc_info.need_nz:
            self.mask_full[:batch_size, :dim_0, :dim_1] = attention_mask.squeeze(1)
        else:
            self.mask_full = torch.zeros((self.batch_num, self.max_position_embeddings,
                self.max_position_embeddings), dtype=self.dtype, device=input_ids_or_embedding.device)
            self.mask_full[:batch_size, :dim_0, :dim_1] = attention_mask.squeeze(1)
            self.mask_full = torch_npu.npu_format_cast_(
                self.mask_full.view(self.batch_num, self.mask_full.shape[1],
                self.mask_full.shape[2] // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)

    @abstractmethod
    def prepare_inputs(self,
                       input_ids_or_embedding: torch.Tensor,
                        position_ids: torch.Tensor,
                        cu_seqlen_prefill: Optional[bool],
                        max_seq_len: int,
                        ):
        """Abstract method for preparing forward pass inputs."""
        pass

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor, optional): The input ids tensor, defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor, defaults to None.
            position_ids (torch.LongTensor, optional): The position ids tensor, defaults to None.
            past_key_values (List[torch.FloatTensor], optional): The past key values tensor, defaults to None.
            inputs_embeds (torch.FloatTensor, optional): The input embedding tensor, defaults to None.
            labels (torch.LongTensor, optional): The labels tensor, defaults to None.
            use_cache (bool, optional): Whether to use cache, defaults to None.
            output_attentions (bool, optional): Whether to output attentions, defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states, defaults to None.
            return_dict (bool, optional): Whether to return as a dict format, defaults to None.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        self.init_kvcache(inputs_embeds if inputs_embeds is not None else input_ids, past_key_values)
        position_ids = self.init_position_ids(inputs_embeds if inputs_embeds is not None else input_ids, position_ids)
        self.init_mask(inputs_embeds if inputs_embeds is not None else input_ids, attention_mask)

        cu_seqlen_prefill = True if not past_key_values else False
        target_key = PREFILL if cu_seqlen_prefill else DECODE
        self.graph_inputs[target_key]["layer_id"] = torch.zeros(1, dtype=torch.int32, device=input_ids.device)
        self.graph_inputs[target_key]["attention_mask"] = self.mask_full
        
        self.prepare_inputs(
            inputs_embeds if inputs_embeds is not None else input_ids,
            position_ids,
            cu_seqlen_prefill,
            self.max_position_embeddings,
        )
        if cu_seqlen_prefill:
            atb_model_out = self.prefill_graph.forward(self.graph_inputs[PREFILL], self.graph_outputs[PREFILL],
                                                       self.graph_param[PREFILL])
        else:
            atb_model_out = self.decode_graph.forward(self.graph_inputs[DECODE], self.graph_outputs[DECODE],
                                                      self.graph_param[DECODE])
        
        logits = atb_model_out[self.get_out_tensor_names()[0]]
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        next_cache = [self.k_cache, self.v_cache] if use_cache else None
        if not return_dict:
            return (loss,) + tuple(v for v in [logits, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(
            self, input_ids: torch.Tensor, past_key_values: torch.Tensor = None,
            attention_mask: torch.Tensor = None, inputs_embeds: torch.Tensor = None, **kwargs
    ) -> dict:
        """
        Prepare inputs for generation.

        Args:
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def get_input_embeddings(self) -> nn.Module:
        """Return the input embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        """Set the input embeddings."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """Return the output embeddings."""
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings: nn.Module):
        """Set the output embeddings."""
        self.lm_head.linear = new_embeddings