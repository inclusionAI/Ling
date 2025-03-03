# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch
from torch import nn

from atb_llm.models.base.modeling import get_suffix, FlashAttention, MLP, FlashLayer

from atb_llm.utils.layers import (
    TensorEmbedding,
    RMSNorm,
    RMSNormBias,
    RMSNormWrapper,
    RMSNormAntiOutlierWrapper
)
from atb_llm.utils.quantize.pack_type import PackType



class GemmaRMSNorm(RMSNorm):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        GemmaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(prefix, weights)

        self.weight = nn.Parameter((weights.get_tensor(f"{prefix}.weight") + 1).contiguous())


class GemmaRMSNormBias(RMSNormBias):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        GemmaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(prefix, weights)

        self.weight = nn.Parameter((weights.get_tensor(f"{prefix}.weight") + 1).contiguous())


class GemmaRMSNormWrapper(RMSNormWrapper):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(prefix, weights)

        self.ori = GemmaRMSNorm(prefix, weights, eps)
        self.anti = GemmaRMSNormBias(f'{prefix}.module', weights, eps)


class GemmaRMSNormAntiOutlierWrapper(RMSNormAntiOutlierWrapper):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        GemmaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(prefix, weights)

        self.ori = GemmaRMSNorm(f'{prefix}.ori', weights, eps)
        self.anti = GemmaRMSNormBias(f'{prefix}.anti', weights, eps)


class GemmaMLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)



class FlashGemmaAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.head_size = config.head_dim
        self.rope_theta = config.rope_theta
        self.load_weights(**kwargs)

    def load_qkv_weights(self, **kwargs):
        super().load_qkv_weights(**kwargs)


class FlashGemmaLayer(FlashLayer):
    def __init__(self, layer_id, config, weights):
        super().__init__(layer_id, config, weights)
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashGemmaAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights
        )
        self.mlp = GemmaMLP(
            prefix=f"{prefix}.mlp",
            config=config,
            weights=weights
        )
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.input_layernorm = GemmaRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.input_layernorm = GemmaRMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.input_layernorm = GemmaRMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            self.input_layernorm = GemmaRMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.post_attention_layernorm = GemmaRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.post_attention_layernorm = GemmaRMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.post_attention_layernorm = GemmaRMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            self.post_attention_layernorm = GemmaRMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')



class FlashGemmaModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashGemmaLayer(
                    layer_id,
                    config,
                    weights
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashGemmaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = GemmaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = GemmaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
