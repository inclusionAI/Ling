from typing import Optional, List, Tuple
import torch
from torch import nn
from atb_llm.models.base.modeling import FlashAttention, MLP
from atb_llm.utils.layers.linear import FastLinear, TensorReplicatedLinear
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    load_column_multi,
    RMSNorm
)
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.quantize.pack_type import get_pack_type
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class DeepseekMLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)

def get_suffix(tensor_name: str) -> str:
    """Get the suffix of a tensor name."""
    return tensor_name.split(".")[-1]

class FlashDeepseekAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_names = [f'{self.prefix}.query_key_value']
        self.dense_name = f'{self.prefix}.dense'
        self.pack_type = get_pack_type(self.weights, self.qkv_names, self.norm_name, self.pack_name)
        '''
        super().load_qkv_weights(**kwargs)

        dense_linear = TensorParallelRowLinear.load_att_dense(
            self.config,
            prefix=self.dense_name,
            weights=self.weights,
            bias=self.dense_bias,
            gqa_size=self.head_size,
            bias_pre_add=self.bias_pre_add
        )
        setattr(self, get_suffix(self.dense_name), dense_linear)
        '''
        if config.model_type == "bailing_moe" and config.num_hidden_layers == 88:
            padding = True

            query_key_value_linear = TensorParallelColumnLinear.load_qkv(
                self.config,
                prefix=self.qkv_names[0],
                weights=self.weights,
                bias=self.qkv_bias,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                padding=padding,
            )
            setattr(self, get_suffix(self.pack_name), query_key_value_linear)

            dense_linear = TensorParallelRowLinear.load_att_dense(
                self.config,
                prefix=self.dense_name,
                weights=self.weights,
                bias=self.dense_bias,
                gqa_size=self.head_size,
                bias_pre_add=self.bias_pre_add
            )
            setattr(self, get_suffix(self.dense_name), dense_linear)
        elif config.model_type == "bailing_moe" and config.num_hidden_layers == 28:
            padding = False
            query_key_value_linear = TensorParallelColumnLinear.load_qkv(
                self.config,
                prefix=self.qkv_names[0],
                weights=self.weights,
                bias=self.qkv_bias,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                padding=padding,
            )
            setattr(self, get_suffix(self.pack_name), query_key_value_linear)

            dense_linear = TensorParallelRowLinear.load(
                self.config,
                prefix=self.dense_name,
                weights=self.weights,
                bias=self.dense_bias,
                gqa_size=self.head_size,
                bias_pre_add=self.bias_pre_add,
            )
            setattr(self, get_suffix(self.dense_name), dense_linear)


class FlashDeepseekLayer(nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                     hidden_states: torch.tensor,
                     residual: torch.tensor,
                     cos: torch.tensor,
                     sin: torch.tensor,
                     cu_seqlen_prefill: torch.tensor,
                     kv_cache: Tuple[torch.tensor, torch.tensor],
                     block_tables: List[torch.tensor],
                     slots: torch.tensor,
                     input_lengths: torch.tensor,
                     max_s: torch.tensor):
            self.hidden_states = hidden_states
            self.residual = residual
            self.cos = cos
            self.sin = sin
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s

    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashDeepseekAttention(
            prefix=f"{prefix}.attention", config=config, weights=weights
        )
        if (config.num_experts is not None and
            layer_id >= config.first_k_dense_replace and
            layer_id % config.moe_layer_freq == 0):
            self.mlp = DeepseekMoE(prefix=f"{prefix}.mlp", config=config, weights=weights, shared_mlp_cls=DeepseekMLP)
        else:
            self.mlp = DeepseekMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.input_layernorm = RMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
            self,
            input_args: ForwardInputArgs
    ):
        hidden_states = input_args.hidden_states
        residual = input_args.residual
        cos = input_args.cos
        sin = input_args.sin
        cu_seqlen_prefill = input_args.cu_seqlen_prefill
        kv_cache = input_args.kv_cache
        block_tables = input_args.block_tables
        slots = input_args.slots
        input_lengths = input_args.input_lengths
        max_s = input_args.max_s
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashDeepseekModel(torch.nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                    input_ids: torch.Tensor,
                    position_ids: torch.Tensor,
                    cu_seqlen_prefill: Optional[torch.Tensor],
                    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                    block_tables: torch.Tensor,
                    slots: torch.Tensor,
                    input_lengths: torch.Tensor,
                    max_s: int,
                    lm_head_indices: Optional[torch.Tensor] = None):
            self.input_ids = input_ids
            self.position_ids = position_ids
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s
            self.lm_head_indices = lm_head_indices

    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="model.word_embeddings", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashDeepseekLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, prefix, config, weights, shared_mlp_cls):
        super().__init__()
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = True # defaulting the model to tensor parallel
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        if self.expert_parallel_degree == 0:
            msg = "expert parallel degree should not be 0!"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        self.expert_lists = []
        if self.tp:
            self.expert_lists = [[i for i in range(config.num_experts)] for j in range(self.tp_world_size)]
        else:
            self.expert_lists = assign(config.num_experts, self.tp_world_size)
        self.config = config
        self.hidden_dim = self.config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        expert_prefix = f"{prefix}.experts"
        self.gate = FastLinear.load(
                prefix=f"{prefix}.gate",
                weights=weights,
                bias=False,
                )
        linear_names = [f'{expert_prefix}.0.up_proj', f'{expert_prefix}.0.gate_proj']
        pack_name = f'{expert_prefix}.0.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.tp:
            if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                self.gate_up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_up_proj.append(load_column_multi(
                        config,
                        prefixes=[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"],
                        weights=weights,
                        head_size=1,
                    ))
            elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
                self.gate_up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_up_proj.append(TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{expert_prefix}.{i}.gate_up_proj",
                    weights=weights,
                    bias=False,
                    ))
            else:
                self.gate_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_proj.append(TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{expert_prefix}.{i}.gate_proj",
                    weights=weights,
                    bias=False,
                    ))
                self.up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.up_proj.append(TensorParallelColumnLinear.load(
                        config,
                        prefix=f"{expert_prefix}.{i}.up_proj",
                        weights=weights,
                        bias=False,
                    ))
            self.down_proj = nn.ModuleList()
            for i in range(self.num_experts):
                self.down_proj.append(TensorParallelRowLinear.load(
                config,
                prefix=f"{expert_prefix}.{i}.down_proj",
                weights=weights,
                bias=False,
                ))
            self.intermediate_size = (
                    (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
            )
        else:
            if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                self.gate_up_proj = nn.ModuleList()
                for i in self.expert_lists[self.tp_rank]:
                    self.gate_up_proj.append(TensorReplicatedLinear.load(
                        config,
                        prefixes=[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"],
                        weights=weights,
                        head_size=1,
                    ))
            self.down_proj = nn.ModuleList()
            for i in self.expert_lists[self.tp_rank]:
                self.down_proj.append(TensorReplicatedLinear.load(
                config,
                prefix=f"{expert_prefix}.{i}.down_proj",
                weights=weights,
                bias=False,
                ))

        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            shared_expert_prefix = f"{prefix}.shared_experts"
            self.shared_experts = shared_mlp_cls(
                prefix=shared_expert_prefix,
                config=config,
                weights=weights,
                intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            device = expert_cache.device
            expert_cache_cpu = expert_cache.cpu()
            expert_cache_cpu.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]).cpu(),
                                             expert_out.cpu(), reduce='sum')
            expert_cache = expert_cache_cpu.to(device=device)
        return expert_cache