# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import torch.distributed
from torch import nn

from atb_llm.models.base.modeling import FlashAttention, MLP, FlashLayer
from atb_llm.utils.layers import TensorEmbedding, RMSNorm
from atb_llm.models.base.config import BaseConfig


class BunnyConfig(BaseConfig):
    model_type = "bunny"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151646,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=21,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen2MLP(MLP):

    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)


class FlashQwen2Attention(FlashAttention):

    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_bias = True
        self.load_weights(**kwargs)


class FlashQwen2Layer(FlashLayer):

    def __init__(self, layer_id, config, weights, model_prefix="model", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashQwen2Attention(
            prefix=f"{self.prefix}.self_attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = Qwen2MLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashBunnyModel(torch.nn.Module):

    def __init__(self, config, weights, model_prefix="model", **kwargs):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.h = nn.ModuleList(
            [
                FlashQwen2Layer(
                    layer_id,
                    config,
                    weights,
                    model_prefix,
                    ** kwargs
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

        self.head_size = self.h[0].self_attn.head_size
        self.num_heads = self.h[0].self_attn.num_heads