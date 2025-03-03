# coding=utf-8
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
""" PyTorch GPTNeoX model."""

import json
import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from configuration_gpt_neox import GPTNeoXConfig
from torch import nn
from torch.nn import CrossEntropyLoss
import torch_npu
from atb_llm.utils.env import ENV
from atb_speed.common.timer import Timer
from atb_speed.common.utils import load_atb_speed
from atb_llm.utils.initial import NPUSocInfo
from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"

GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neox-20b",
]


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


IS_ND = is_nd()
logger.info(f"IS_ND = {IS_ND}")


def get_rank_and_world_size():
    try:
        rank = ENV.rank
        world_size = ENV.world_size
    except RuntimeError:
        rank = 0
        world_size = 1
    return rank, world_size


RANK, WORLD_SIZE = get_rank_and_world_size()
logger.info(f"RANK = {RANK} | WORLD_SIZE = {WORLD_SIZE}")


def load_ascend_transformer():
    atb_speed_home_path = ENV.atb_speed_home_path
    lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
    logger.info(f"load {lib_path}")
    torch.classes.load_library(lib_path)


load_ascend_transformer()
load_atb_speed()


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), 1, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, past_key_values_length + tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask


def _prepare_input_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    combined_attention_mask = None
    if input_shape[-1] > 1:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        if combined_attention_mask is None:
            combined_attention_mask = expanded_attn_mask
        else:
            combined_attention_mask = combined_attention_mask + expanded_attn_mask

    return combined_attention_mask.masked_fill(combined_attention_mask.to(torch.bool), -20000)


class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
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


class GPTNeoXAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.world_size = WORLD_SIZE
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_attention_heads = self.num_attention_heads // self.world_size

        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings,
            base=config.rotary_emb_base
        )
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size // self.world_size)
        self.dense = nn.Linear(config.hidden_size // self.world_size, config.hidden_size)

        self.layer_id = layer_id

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor,
            head_mask: Optional[torch.FloatTensor] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        qkv = self.query_key_value(hidden_states)

        # Covert QKV to multiHead shape
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size: 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        # all reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(
                attn_output, op=torch.distributed.ReduceOp.SUM)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        positions = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # out product of positions and inv_freq: [max_seq_len_cached, dim//2]
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            positions = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            # out product of positions and inv_freq: [seq_len, dim//2]
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, ...].to(x.device), self.sin_cached[:, :, :seq_len, ...].to(x.device)


class AscendRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__(dim, max_position_embeddings, base, device)
        self.cos_cached = self.cos_cached.squeeze(1).squeeze(0).half()
        self.sin_cached = self.sin_cached.squeeze(1).squeeze(0).half()

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            super().forward(x, seq_len)
            self.cos_cached = self.cos_cached.squeeze(1).squeeze(0).half()
            self.sin_cached = self.sin_cached.squeeze(1).squeeze(0).half()
        if x.device != self.cos_cached.device:
            self.cos_cached = self.cos_cached.to(x.device).half()
            self.sin_cached = self.sin_cached.to(x.device).half()
        return self.cos_cached, self.sin_cached


class AttentionMask(nn.Module):
    def __init__(self, max_seq_length):
        super().__init__()
        self.mask_min = -20000
        self._seq_len_cached = max_seq_length
        self.attn_mask_inc_cache = torch.full((max_seq_length, max_seq_length), self.mask_min, dtype=torch.half).npu()
        self.attn_mask_inc_zeros = torch.full((max_seq_length, max_seq_length), 0, dtype=torch.half).npu()
        self.attn_mask_full = None  # encoder_mask
        self.attn_mask_inc = None  # decoder_mask

    def get_attn_mask(self, attention_mask, origin_inputs_count, seq_len: int, batch_size,
                      dtype: torch.dtype, device: torch.device):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)

        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            self.attn_mask_inc_cache = torch.full(
                (seq_len, seq_len), self.mask_min).to(dtype).to(device)
            self.attn_mask_inc_zeros = torch.full((seq_len, seq_len), 0).to(dtype).to(device)
        if self.attn_mask_inc_cache.device != device or self.attn_mask_inc_cache.dtype != dtype:
            self.attn_mask_inc_cache = self.attn_mask_inc_cache.to(dtype).to(device)
            self.attn_mask_inc_zeros = self.attn_mask_inc_zeros.to(dtype).to(device)

        self.attn_mask_full = torch.full(
            (batch_size, self._seq_len_cached, self._seq_len_cached), self.mask_min).to(dtype).to(device)
        decoder_masks = []
        for i in range(batch_size):
            self.attn_mask_full[i][:seq_len, :seq_len] = attention_mask.squeeze(1)[i]
            count = origin_inputs_count[i].item()
            # left padding, if input has no padding，count will equal seq_len
            left_mask = self.attn_mask_inc_cache[:, :seq_len - count]
            right_mask = self.attn_mask_inc_zeros[:, :self._seq_len_cached - seq_len + count]
            decoder_mask = torch.concat([left_mask, right_mask], dim=-1).unsqueeze(0)

            decoder_masks.append(decoder_mask)
        self.attn_mask_inc = torch.concat(decoder_masks, dim=0).to(dtype).to(device)

        return self.attn_mask_full, self.attn_mask_inc


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k: [bs, num_heads, seq_len, rotary_ndims]
    # cos, sin: [1, 1, seq_len, rotary_ndims]
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])  # [bs, 1, seq_len, rotary_ndims]
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.world_size = WORLD_SIZE
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size // self.world_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size // self.world_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        # all_reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(
                hidden_states, op=torch.distributed.ReduceOp.SUM)
        return hidden_states


class GPTNeoXLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.world_size = WORLD_SIZE
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config, layer_id)
        self.mlp = GPTNeoXMLP(config)
        self.layer_id = layer_id

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor],
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
    ):
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


GPT_NEOX_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module] sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~GPTNeoXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEOX_INPUTS_DOCSTRING = r"""
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
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
        self.config = config
        self.rank = RANK
        self.rank_size = WORLD_SIZE
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GPTNeoXLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # for ascend init
        self.acl_param_decoder = None
        self.acl_param_encoder = None
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.ascend_weight = []
        self.init_ascend_operations(config)
        self.place_holder = torch.ones(1).npu()
        self.hidden_size_nz = None
        self.soc_info = None

    def init_ascend_operations(self, config: GPTNeoXConfig):
        # for soc info
        self.soc_info = NPUSocInfo()
        try:
            self.communication_backend = self.soc_info.communication_backend
        except AttributeError:
            self.communication_backend = "hccl" # hccl, lccl

        self.acl_param_encoder = json.dumps({
            "headNum": config.num_attention_heads // self.rank_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "layerNormEps": config.layer_norm_eps,
            "rotaryPct": config.rotary_pct,
            "isPrefill": True,
            "qScale": 1 / math.sqrt(config.hidden_size // config.num_attention_heads),
            "rank": self.rank,
            "rankSize": self.rank_size,
            "backend": self.communication_backend,
        })

        self.acl_param_decoder = json.dumps({
            "headNum": config.num_attention_heads // self.rank_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "layerNormEps": config.layer_norm_eps,
            "rotaryPct": config.rotary_pct,
            "isPrefill": True,
            "qScale": 1 / math.sqrt(config.hidden_size // config.num_attention_heads),
            "rank": self.rank,
            "rankSize": self.rank_size,
            "backend": self.communication_backend,
        })

        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads // self.rank_size

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("gptneox_20b_FaKvCacheRopeModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("gptneox_20b_FaKvCacheRopeModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        rotary_dim = int((config.hidden_size // config.num_attention_heads) * config.rotary_pct)
        self.ascend_rotary_embedding = AscendRotaryEmbedding(
            rotary_dim, max_position_embeddings=self.max_position_embeddings)

        self.increment_flags = [False] * self.num_layers
        self.token_num = 0

        self.token_offset = None
        self.layer_id_input = []

        self.attention_mask_generator = AttentionMask(self.max_position_embeddings)
        self.attention_mask_input = None
        self.attention_mask_encoder = None
        self.attention_mask_decoder = None
        self.origin_inputs_count = None

        self.seq_len_tensor = None
        self.seqlen_max = None

        for i in range(self.num_layers):
            self.layer_id_input.append(torch.tensor([i], dtype=torch.int32).npu())

        self.weight_flag = False
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.nz_dim = 16

        self.acl_encoder_operation_inputs: list = [None] * (11 + self.num_layers)
        self.acl_decoder_operation_inputs: list = [None] * (11 + self.num_layers)
        self.lm_head_weight = None
        self.k_cache_input = None
        self.v_cache_input = None
        self.batch = 0

        for i in range(self.num_layers):
            self.acl_encoder_operation_inputs[11 + i] = torch.tensor([i], dtype=torch.int32).npu()
            self.acl_decoder_operation_inputs[11 + i] = torch.tensor([i], dtype=torch.int32).npu()

    def init_ascend_weight(self):
        weights = [self.state_dict()["embed_in.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.layers[i].state_dict()
            weights_t.append(weights_layer['input_layernorm.weight'])
            weights_t.append(weights_layer['input_layernorm.bias'])
            weights_t.append(weights_layer['post_attention_layernorm.weight'])
            weights_t.append(weights_layer['post_attention_layernorm.bias'])
            weights_t.append(weights_layer['attention.query_key_value.weight'])
            weights_t.append(weights_layer['attention.query_key_value.bias'])
            weights_t.append(weights_layer['attention.dense.weight'])
            weights_t.append(weights_layer['attention.dense.bias'])
            weights_t.append(weights_layer['mlp.dense_h_to_4h.weight'])
            weights_t.append(weights_layer['mlp.dense_h_to_4h.bias'])
            weights_t.append(weights_layer['mlp.dense_4h_to_h.weight'])
            weights_t.append(weights_layer['mlp.dense_4h_to_h.bias'])
            weights.extend(weights_t)

        weights.append(self.state_dict()["final_layer_norm.weight"])
        weights.append(self.state_dict()["final_layer_norm.bias"])
        weights.append(self.lm_head_weight)

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

    def prepare_inputs_for_ascend(self, input_ids, position_ids, seq_length, batch_size, past_key_values=None):
        max_seq_len = self.token_num + seq_length
        cos_table, sin_table = self.ascend_rotary_embedding(input_ids, max_seq_len)
        if not past_key_values or past_key_values[0] is None:
            self.token_num = seq_length
            self.token_offset[:] = seq_length
            self.seq_len_tensor = torch.tensor([seq_length] * batch_size,
                                               dtype=torch.int32, device=input_ids.device)
            self.seqlen_max = torch.tensor([self.seq_len_tensor[0] - 1], dtype=torch.int64, device="npu")
            self.attention_mask_encoder, self.attention_mask_decoder = self.attention_mask_generator.get_attn_mask(
                self.attention_mask_input,
                self.origin_inputs_count,
                seq_length,
                batch_size,
                dtype=self.k_cache_input.dtype,
                device=self.k_cache_input.device)

            if not IS_ND:
                self.attention_mask_encoder = torch_npu.npu_format_cast(self.attention_mask_encoder.view(
                    batch_size, self.max_position_embeddings,
                    self.max_position_embeddings // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)
                self.attention_mask_decoder = torch_npu.npu_format_cast(self.attention_mask_decoder.view(
                    batch_size, self.max_position_embeddings,
                    self.max_position_embeddings // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)

            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids
            self.acl_encoder_operation_inputs[2] = cos_table
            self.acl_encoder_operation_inputs[3] = sin_table
            self.acl_encoder_operation_inputs[4] = self.attention_mask_encoder
            self.acl_encoder_operation_inputs[5] = self.k_cache_input
            self.acl_encoder_operation_inputs[6] = self.v_cache_input
            self.acl_encoder_operation_inputs[7] = self.token_offset
            self.acl_encoder_operation_inputs[8] = self.seq_len_tensor
            self.acl_encoder_operation_inputs[9] = self.place_holder
            self.acl_encoder_operation_inputs[10] = self.seqlen_max

            acl_param_encoder = json.dumps({
                "tokenOffset": [seq_length] * batch_size,
                "seqLen": [seq_length] * batch_size
            })

            return self.acl_encoder_operation_inputs, acl_param_encoder
        else:
            self.token_num = self.token_num + 1
            self.token_offset[:] = self.token_num
            self.seq_len_tensor = torch.tensor([1] * batch_size, dtype=torch.int32, device=input_ids.device)
            self.seqlen_max = torch.tensor([self.seq_len_tensor[0] - 1], dtype=torch.int64, device="npu")

            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids
            self.acl_decoder_operation_inputs[2] = cos_table
            self.acl_decoder_operation_inputs[3] = sin_table
            self.acl_decoder_operation_inputs[4] = self.attention_mask_decoder
            self.acl_decoder_operation_inputs[5] = self.k_cache_input
            self.acl_decoder_operation_inputs[6] = self.v_cache_input
            self.acl_decoder_operation_inputs[7] = self.token_offset
            self.acl_decoder_operation_inputs[8] = self.seq_len_tensor
            self.acl_decoder_operation_inputs[9] = self.place_holder
            self.acl_decoder_operation_inputs[10] = self.seqlen_max

            acl_param_decoder = json.dumps({
                "tokenOffset": [self.token_num] * batch_size,
                "seqLen": [1] * batch_size
            })

            return self.acl_decoder_operation_inputs, acl_param_decoder

    def execute_ascend_operator(self, acl_model, input_ids, position_ids, past_key_values=None):
        batch_size, seq_length = input_ids.shape
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, seq_length, batch_size,
                                                               past_key_values)
        acl_model_out = acl_model.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if batch_size != self.batch:
            self.batch = batch_size

            if not IS_ND:
                self.hidden_size_nz = math.ceil(self.hidden_size // self.rank_size / self.nz_dim)
                self.k_cache_input = torch.zeros(self.num_layers,
                                                 batch_size,
                                                 self.hidden_size_nz,
                                                 self.max_position_embeddings,
                                                 self.nz_dim,
                                                 device=input_ids.device,
                                                 dtype=torch.half)
                self.v_cache_input = torch.zeros(self.num_layers,
                                                 batch_size,
                                                 self.hidden_size_nz,
                                                 self.max_position_embeddings,
                                                 self.nz_dim,
                                                 device=input_ids.device,
                                                 dtype=torch.half)
                self.k_cache_input = torch_npu.npu_format_cast(self.k_cache_input, 29)
                torch.npu.empty_cache()
                self.v_cache_input = torch_npu.npu_format_cast(self.v_cache_input, 29)
            else:

                self.k_cache_input = torch.zeros(self.num_layers,
                                                 batch_size,
                                                 self.max_position_embeddings,
                                                 self.hidden_size // self.rank_size,
                                                 device=input_ids.device,
                                                 dtype=torch.half)
                self.v_cache_input = torch.zeros(self.num_layers,
                                                 batch_size,
                                                 self.max_position_embeddings,
                                                 self.hidden_size // self.rank_size,
                                                 device=input_ids.device,
                                                 dtype=torch.half)
            torch.npu.empty_cache()
            self.token_num = 0
            self.token_offset = torch.full((batch_size,), 0, dtype=torch.int32, device=self.k_cache_input.device)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
            if attention_mask is None:
                seq_length_with_past = seq_length + past_length
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=input_ids.device)

            self.origin_inputs_count = attention_mask.sum(dim=-1)
            self.attention_mask_input = _prepare_input_attention_mask(
                attention_mask, (batch_size, seq_length), self.k_cache_input, past_length)
        else:
            past_length = self.token_num

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if not self.ascend_weight:
            self.init_ascend_weight()

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        is_prefill = True if past_key_values[0] is None else False

        if is_prefill:
            model_op = self.acl_encoder_operation
        else:
            model_op = self.acl_decoder_operation

        hidden_states = self.execute_ascend_operator(model_op,
                                                     input_ids,
                                                     position_ids,
                                                     past_key_values)

        presents = (self.k_cache_input, self.v_cache_input)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    """GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.""", GPT_NEOX_START_DOCSTRING
)
class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        else:
            self.world_size = WORLD_SIZE
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size // self.world_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.position_ids_cache = torch.arange(int(config.max_position_embeddings),
                                               device='cpu').long().unsqueeze(0).npu()
        self.lm_head_weight = None

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @torch.no_grad()
    @Timer.timing
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:
        """
        if self.lm_head_weight is None:
            self.lm_head_weight = self.state_dict()["embed_out.weight"]
            if not IS_ND:
                self.lm_head_weight = torch_npu.npu_format_cast(self.lm_head_weight, 29)
            self.gpt_neox.lm_head_weight = self.lm_head_weight

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = outputs[0]

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        input_shape = input_ids.shape
        if input_shape[0] != self.position_ids_cache.shape[0]:
            self.position_ids_cache = self.position_ids_cache.repeat(input_shape[0], 1)

        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = self.position_ids_cache[:, :input_shape[1]]

        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
