# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Optional, List, Tuple

import torch

from .modeling_qwen2_atb import QwenModelATB
from ..base.flash_causal_lm_atb import FlashForCausalLMATB, PREFILL, DECODE
from ...utils.layers import load_column_multi, TensorHead
from ...common_op_builders.data_type import CommonOpBuilderType
from ...common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from ...common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend


class FlashQwen2ForCausalLMATB(FlashForCausalLMATB):
    def __init__(self, config, weights, lm_head_prefix="lm_head", model_prefix="model", **kwargs):
        super().__init__(config, weights, **kwargs)
        # 模型结构相关
        self.backend = CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL
        self.model_prefix = model_prefix
        self.model = QwenModelATB(config, weights, model_prefix, lm_head_prefix,
                                is_fa=False, speculate_enable=self.speculate_enable, backend=self.backend)
        self.final_norm_prefix = f"{model_prefix}.norm"
        self.lm_head_prefix = lm_head_prefix
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=lm_head_prefix,
                weights=weights,
                is_norm=False,
            )
        else:
            if config.tie_word_embeddings:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=["model.embed_tokens"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
            else:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[lm_head_prefix],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    @property
    def name(self):
        return "qwen"

    def get_in_tensor_names(self, is_prefill):
        default_input = [
            'input_ids', 'position_ids', 'cos_table', 'sin_table',
            'slots_mapping', 'seq_len', 'attention_mask'
        ]
        if is_prefill:
            default_input.extend(['lm_head_indices'])
        else:
            default_input.extend(['block_tables'])
            if self.speculate_enable:
                default_input.extend(['attention_mask', 'q_len'])
        return default_input

    def get_out_tensor_names(self):
        return ['model_out']

    def build_graph(self, graph, is_prefill):
        # 设置输入输出
        kv_cache_names = []
        for i in range(self.config.num_hidden_layers):
            kv_cache_names.extend([f"layer_{i}_k_cache", f"layer_{i}_v_cache"])
        graph.add_input_output(
            input=list(self.weight.keys()) + kv_cache_names + self.get_in_tensor_names(is_prefill),
            output=self.get_out_tensor_names())

        # 增加图节点
        self.model.build_graph(graph, is_prefill)

        op_name = 'op_name'
        category = 'category'

        lm_head_linear_param = {
            op_name: "lm_head_linear",
            category: CommonOpBuilderType.LINEAR,
            "linear_module": self.lm_head.linear,
            "default_dtype": self.dtype,
        }
        lm_head_linear_parallel_param = {
            op_name: "lm_head_linear_parallel",
            category: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_GATHER,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend),
            "linear_param": lm_head_linear_param,
        }
        lm_head_param = {
            op_name: "test_lm_head",
            category: CommonOpBuilderType.LM_HEAD,
            "enable_linear_parallel": True,
            "linear_parallel_param": lm_head_linear_parallel_param,
            "gather_ahead": True if is_prefill else False,
            "unpad_inputs": True
        }
        lm_head_linear_tensor_map = {
            "input": f"{self.final_norm_prefix}_out",
            "indices": "lm_head_indices",
            "linear_out": self.get_out_tensor_names()[0]
        }
        lm_head_builder = CommonOpBuilderManager.get_builder(lm_head_param)
        graph = lm_head_builder.build(graph, lm_head_linear_tensor_map)

        # 构图
        graph.execute_as_single = False
        graph.build()


    def prepare_inputs(
            self, input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # 准备输入
        # q lens
        q_lens = kwargs.get('q_lens', [])
        # attention mask
        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            if self.soc_info.need_nz:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        else:
            attention_mask = self.attn_mask.get_attn_mask(1, dtype=self.dtype, device="npu")
            if self.speculate_enable:
                q_lens = kwargs.get('q_lens', [])
                spec_mask = kwargs.get('spec_mask', None)
                q_lens = torch.tensor(q_lens).to(self.device).to(torch.int32)
                req_mask = spec_mask
                if self.soc_info.need_nz:
                    req_mask = self.transdata_operation.execute([req_mask])[0]
                attention_mask = req_mask
        # 更新输入
        target_key = PREFILL if is_prefill else DECODE
        self.graph_inputs[target_key].update({
            "input_ids": input_ids,
            "position_ids": position_ids.to(torch.int64),
            "slots_mapping": slots.to(torch.int32),
            "seq_len": input_lengths.to(torch.int32),
            "cos_table": self.cos_embed,
            "sin_table": self.sin_embed
        })
        if attention_mask is not None:  # attention mask
            self.graph_inputs[target_key]["attention_mask"] = attention_mask
        if is_prefill and lm_head_indices is None:  # lm head indices
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                           dtype=torch.int64, device=input_ids.device)
        if is_prefill:
            self.graph_inputs[target_key]["lm_head_indices"] = lm_head_indices
        else:  # decode
            self.graph_inputs[target_key]["block_tables"] = block_tables.to(torch.int32)
            if self.speculate_enable:
                q_len = torch.tensor(q_lens, dtype=torch.int32, device=self.device)
                self.graph_inputs[target_key]["q_len"] = q_len

        # 准备输出
        real_vocab_size = self.weight.get(f"{self.lm_head_prefix}.weight").shape[0] * self.tp_world_size
        batch_size = lm_head_indices.shape[0] if is_prefill else input_ids.shape[0]

        self.graph_outputs[target_key][self.get_out_tensor_names()[0]] = \
            torch.ones(batch_size, real_vocab_size, dtype=self.dtype, device=self.device)

        # 准备bind tensor
        self.graph_param[target_key]['seq_len'] = input_lengths.cpu().to(torch.int32)

        if self.speculate_enable and not is_prefill:
            q_len = torch.tensor(q_lens, dtype=torch.int32, device=self.device)
            self.graph_param[target_key]['q_len'] = q_len.cpu()

