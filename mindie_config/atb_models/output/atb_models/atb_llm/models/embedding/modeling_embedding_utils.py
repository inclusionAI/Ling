# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT and XLM-RoBERTa flash attention model common utils."""

import json
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch_npu
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig

from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.log.logging import logger


_ATB_MODEL_NAME = "bert_EncoderModel"
_ATB_NPU_DEVICE = "npu"
NEED_ND = not NPUSocInfo().need_nz
MASK_INC_DIM1 = 1 if NEED_ND else 16


def create_position_ids_from_input_ids(
        input_ids: torch.Tensor,
        padding_idx: Optional[int] = None,
        past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def create_position_ids_from_inputs_embeds(
        inputs_embeds: torch.Tensor,
        padding_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
    """
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(
        padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
    )
    return position_ids.unsqueeze(0).expand(input_shape)


class KVAttentionManager:
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig], batch_size: int) -> None:
        self.batch_size = batch_size
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_position_embeddings - (2 if isinstance(config, XLMRobertaConfig) else 0)
        self.num_layers = config.num_hidden_layers

        self.dummy_size = 1
        self.is_full = True
        self.max_attention_mask_len = self.max_seq_len  # 非倒三角矩阵，不支持 maskfree 特性
        self.min_cache = None
        self.nz_dim = 16
        self.ori_len_list = []
        self.token_offset = 1

        self.k_cache_input = torch.zeros(
            self.dummy_size,
            device=_ATB_NPU_DEVICE,
            dtype=torch.float16
        )
        self.v_cache_input = torch.zeros(
            self.dummy_size,
            device=_ATB_NPU_DEVICE,
            dtype=torch.float16
        )

        self.attention_mask_max_full = torch.zeros(
            (self.batch_size, self.max_attention_mask_len, self.max_attention_mask_len),
            device=_ATB_NPU_DEVICE,
            dtype=torch.float16
        )
        self.attention_mask_max_inc = torch.zeros(
            (self.batch_size, MASK_INC_DIM1, self.max_attention_mask_len),
            device=_ATB_NPU_DEVICE,
            dtype=torch.float16
        )

        # init attributes in self.init_seq_len_and_token_offset()
        self.token_offset_tensor = None
        self.seq_len_tensor_inc = None
        self.seq_len_list_inc = None
        self.seq_len_tensor_full = None
        self.seq_len_list_full = None

    @property
    def seq_len_list(self) -> List[int]:
        if self.is_full:
            return self.seq_len_list_full
        return self.seq_len_list_inc

    @property
    def seq_len_tensor(self) -> torch.Tensor:
        if self.is_full:
            return self.seq_len_tensor_full
        return self.seq_len_tensor_inc

    @property
    def token_offset_list(self) -> List[int]:
        return [self.token_offset] * self.batch_size

    def init_attention_mask(self) -> None:
        if NEED_ND:
            self.attention_mask_max_full.zero_()
            self.attention_mask_max_inc.zero_()
        else:
            self.attention_mask_max_full.zero_()
            self.attention_mask_max_inc = torch.zeros(
                (self.batch_size, MASK_INC_DIM1, self.max_attention_mask_len),
                device=_ATB_NPU_DEVICE,
                dtype=torch.float16
            )

    def get_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.is_full:
            for i in range(self.batch_size):
                self.attention_mask_max_full[i][:self.token_offset, :self.token_offset] = attention_mask[i]
            if NEED_ND:
                return self.attention_mask_max_full
            else:
                self.attention_mask_max_inc = self.trans_data(self.attention_mask_max_inc, "inc")
                return self.trans_data(self.attention_mask_max_full, "full")
        else:
            return self.attention_mask_max_inc

    def init_seq_len_and_token_offset(self, seq_len: int) -> None:
        self.token_offset = seq_len
        self.seq_len_list_full = [self.token_offset] * self.batch_size
        self.seq_len_tensor_full = torch.full(
            (self.batch_size,),
            self.token_offset,
            device=_ATB_NPU_DEVICE,
            dtype=torch.int32
        )
        self.seq_len_list_inc = [1] * self.batch_size
        self.seq_len_tensor_inc = torch.full(
            (self.batch_size,),
            1,
            device=_ATB_NPU_DEVICE,
            dtype=torch.int32
        )
        self.token_offset_tensor = torch.full(
            (self.batch_size,),
            self.token_offset,
            device=_ATB_NPU_DEVICE,
            dtype=torch.int32
        )

    def trans_data(self, tensor: torch.Tensor, trans_type: str = "full") -> torch.Tensor:
        if trans_type == "full":
            return torch_npu.npu_format_cast(
                tensor.view(
                    self.batch_size, self.max_seq_len, self.max_seq_len // self.nz_dim, self.nz_dim
                ).transpose(1, 2).contiguous(),
                29
            )
        else:
            return torch_npu.npu_format_cast(
                tensor.view(
                    self.batch_size, self.nz_dim, self.max_seq_len // self.nz_dim, self.nz_dim
                ).transpose(1, 2).contiguous(),
                29
            )


class BaseEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False if isinstance(config, BertConfig) else True
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False
        )

        if isinstance(config, XLMRobertaConfig):
            self.padding_idx = config.pad_token_id
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            if isinstance(self.config, BertConfig):
                position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
            elif isinstance(self.config, XLMRobertaConfig):
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(
                        input_ids,
                        self.padding_idx,
                        past_key_values_length
                    )
                else:
                    position_ids = create_position_ids_from_inputs_embeds(inputs_embeds, self.padding_idx)
            else:
                raise TypeError("config type should be BertConfig or XLMRobertaConfig")

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros,
        # which usually occurs when its auto-generated,
        # registered buffer helps users when tracing the model without passing token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BaseSelfAttention(nn.Module):
    def __init__(
            self,
            config: Union[BertConfig, XLMRobertaConfig],
            position_embedding_type: Optional[str] = None
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        try:
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        except ZeroDivisionError:
            raise
        except Exception as e:
            raise RuntimeError from e
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        try:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        except ZeroDivisionError:
            raise
        except Exception as e:
            raise RuntimeError from e
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in XLMRobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BaseSelfOutput(nn.Module):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BaseAttention(nn.Module):
    def __init__(
            self,
            config: Union[BertConfig, XLMRobertaConfig],
            position_embedding_type: Optional[str] = None
    ) -> None:
        super().__init__()
        self.self = BaseSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BaseSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: List[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BaseIntermediate(nn.Module):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BaseOutput(nn.Module):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BaseLayer(nn.Module):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BaseAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BaseAttention(config, position_embedding_type="absolute")
        self.intermediate = BaseIntermediate(config)
        self.output = BaseOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            present_key_value = 0

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BaseEncoder(nn.Module):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BaseLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[Union[Tuple[torch.Tensor], torch.Tensor], ...], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if isinstance(self.config, XLMRobertaConfig):
                    def create_custom_forward(module, past_key_value_tensor, output_attentions_tensor):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value_tensor, output_attentions_tensor)
                        return custom_forward
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module, past_key_value, output_attentions),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            model_output_with_past_and_cross_attentions = [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            return tuple(
                v
                for v in model_output_with_past_and_cross_attentions
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BasePooler(nn.Module):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BasePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def _init_weights(self, module: Callable) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


BASE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [`torch.nn.Module`] subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BaseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BASE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BaseModel(BasePreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need] by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__(config)
        self.config = config

        # Initialize Ascend parameters
        self.ascend_weight = []
        self.batch_size = 0
        self.head_num = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.kv_attention_manager = None
        self.layer_id_list = [
            torch.tensor([i], device=_ATB_NPU_DEVICE, dtype=torch.int32)
            for i in range(config.num_hidden_layers)
        ]
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers

        # Initialize model
        if hasattr(config, "world_size"):
            rank = torch.distributed.get_rank()
            rank_size = torch.distributed.get_world_size()
            self.acl_param = json.dumps({
                "dk": self.head_size,
                "geluApproximate": -1,
                "headNum": self.head_num,
                "layerNormEps": self.config.layer_norm_eps,
                "layerNormImplMode": 1,
                "layerNum": self.num_layers,
                "rank": rank,
                "rankSize": rank_size
            })
        else:
            self.acl_param = json.dumps({
                "dk": self.head_size,
                "geluApproximate": -1,
                "headNum": self.head_num,
                "layerNormEps": self.config.layer_norm_eps,
                "layerNormImplMode": 1,
                "layerNum": self.num_layers,
            })
        self.model = torch.classes.ModelTorch.ModelTorch(_ATB_MODEL_NAME)
        self.model.set_param(self.acl_param)

    def init_ascend_weight(self) -> None:
        weights: List = [
            self.state_dict()["embeddings.word_embeddings.weight"],
            self.state_dict()["embeddings.position_embeddings.weight"],
            self.state_dict()["embeddings.token_type_embeddings.weight"],
            self.state_dict()["embeddings.LayerNorm.weight"],
            self.state_dict()["embeddings.LayerNorm.bias"]
        ]
        for i in range(self.num_layers):
            weights_per_layer: List = []
            weights_layer = self.encoder.layer[i].state_dict()
            weights_per_layer.append(weights_layer["attention.self.query.weight"])
            weights_per_layer.append(weights_layer["attention.self.query.bias"])
            weights_per_layer.append(weights_layer["attention.self.key.weight"])
            weights_per_layer.append(weights_layer["attention.self.key.bias"])
            weights_per_layer.append(weights_layer["attention.self.value.weight"])
            weights_per_layer.append(weights_layer["attention.self.value.bias"])
            weights_per_layer.append(weights_layer["attention.output.dense.weight"])
            weights_per_layer.append(weights_layer["attention.output.dense.bias"])
            weights_per_layer.append(weights_layer["attention.output.LayerNorm.weight"])
            weights_per_layer.append(weights_layer["attention.output.LayerNorm.bias"])
            weights_per_layer.append(weights_layer["intermediate.dense.weight"])
            weights_per_layer.append(weights_layer["intermediate.dense.bias"])
            weights_per_layer.append(weights_layer["output.dense.weight"])
            weights_per_layer.append(weights_layer["output.dense.bias"])
            weights_per_layer.append(weights_layer["output.LayerNorm.weight"])
            weights_per_layer.append(weights_layer["output.LayerNorm.bias"])
            weights.extend(weights_per_layer)

        self.ascend_weight = weights
        self.model.set_weight(weights)

    def prepare_inputs_for_ascend(
            self,
            input_ids: Optional[torch.Tensor],
            position_ids: Optional[torch.Tensor],
            token_type_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        self.kv_attention_manager.is_full = not past_key_values
        position_ids = position_ids.npu()
        token_type_ids = token_type_ids.npu()
        attention_mask = attention_mask.float().half()

        if isinstance(self.config, BertConfig):
            attention_mask = attention_mask.clone()
            attention_mask[attention_mask == 0] = -65504.0
            attention_mask[attention_mask == 1] = -0.0

        inputs = [
            input_ids,
            position_ids,
            token_type_ids,
            self.kv_attention_manager.get_attention_mask(attention_mask),
            self.kv_attention_manager.k_cache_input,
            self.kv_attention_manager.v_cache_input,
            self.kv_attention_manager.token_offset_tensor,
            self.kv_attention_manager.seq_len_tensor,
        ] + self.layer_id_list

        return inputs

    def execute_ascend_operator(
            self,
            input_ids: Optional[torch.Tensor],
            position_ids: Optional[torch.Tensor],
            token_type_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[torch.Tensor] = None
    ):
        acl_inputs = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
            past_key_values
        )
        tmp_param = json.dumps({
            "tokenOffset": self.kv_attention_manager.token_offset_list,
            "seqLen": self.kv_attention_manager.seq_len_list
        })
        acl_model_out = self.model.execute(acl_inputs, tmp_param)
        return acl_model_out

    def _forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**. past_key_values (`tuple(
            tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(
            batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and value
            hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.kv_attention_manager = KVAttentionManager(self.config, batch_size)

        # past_key_values_length
        if past_key_values:
            past_key_values_length = self.kv_attention_manager.token_offset
            self.kv_attention_manager.token_offset += 1
            self.kv_attention_manager.token_offset_tensor += 1
        else:
            past_key_values_length = 0
            self.kv_attention_manager.init_attention_mask()
            self.kv_attention_manager.init_seq_len_and_token_offset(seq_length)

        if position_ids is None:
            if isinstance(self.config, BertConfig):
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            elif isinstance(self.config, XLMRobertaConfig):
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(
                        input_ids,
                        self.config.pad_token_id,
                        past_key_values_length
                    )
                else:
                    position_ids = create_position_ids_from_inputs_embeds(inputs_embeds)
            else:
                raise TypeError("config type should be BertConfig or XLMRobertaConfig")
        else:
            if isinstance(self.config, BertConfig):
                position_ids = position_ids.view(-1, seq_length).long()

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        if not past_key_values:
            self.kv_attention_manager.ori_len_list = attention_mask.sum(dim=-1)
            if isinstance(self.config, XLMRobertaConfig):
                # 使用 FA 时，在计算首token时会同时计算增量额 attention_mask
                attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if not self.ascend_weight:
            self.init_ascend_weight()

        encoder_outputs = self.execute_ascend_operator(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
            past_key_values
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

    def _prune_heads(self, heads_to_prune: Dict) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


class BaseForSequenceClassification(BasePreTrainedModel):
    def __init__(self, config: Union[BertConfig, XLMRobertaConfig]) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

    def _calc_loss(self, logits, labels: Optional[torch.Tensor] = None):
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return loss
