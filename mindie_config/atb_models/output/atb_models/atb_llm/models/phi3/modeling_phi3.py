# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.

from torch import nn
from atb_llm.models.base.modeling import MLP, FlashLayer, FlashAttention
from atb_llm.utils.layers import RMSNorm, TensorEmbedding


class Phi3MLP(MLP):
    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)
        self.gate_up_names = [f'{self.prefix}.gate_up_proj']
        
        self.load_weights()
        

class FlashPhi3Attention(FlashAttention):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__(prefix, config, weights)
        self.qkv_names = [f'{self.prefix}.qkv_proj']
        self.hidden_size = config.num_hidden_layers

        self.load_weights()


class FlashPhi3Layer(FlashLayer):
    def __init__(self, layer_id, config, weights):
        super().__init__(layer_id, config, weights)
        
        self.self_attn = FlashPhi3Attention(
            prefix=f"{self.prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = Phi3MLP(prefix=f"{self.prefix}.mlp", config=config, weights=weights)
        self.load_weights()


class FlashPhi3Model(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashPhi3Layer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
