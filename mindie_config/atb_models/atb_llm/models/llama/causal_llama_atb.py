# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import torch

from .modeling_llama_atb import LlamaModelATB
from ..base.causal_lm_atb import CausalLMATB, PREFILL, DECODE
from ...common_op_builders.data_type import CommonOpBuilderType
from ...common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from ...common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from ...utils.layers import load_column_multi
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


class LlamaForCausalLMATB(CausalLMATB):
    def __init__(self, config, weights, lm_head_prefix="lm_head", model_prefix="model", **kwargs):
        super().__init__(config, weights, **kwargs)

        self.model_prefix = model_prefix
        self.backend = CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL
        self.model = LlamaModelATB(config, weights, model_prefix, lm_head_prefix, is_fa=True, backend=self.backend)
        self.final_norm_prefix = f"{model_prefix}.norm"
        self.lm_head_prefix = lm_head_prefix
        self.lm_head = load_column_multi(
            config,
            prefixes=[lm_head_prefix],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.is_fa = True
        self.config = config
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int32).npu()

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.position_embedding_type = config.pe_type
        if self.position_embedding_type != "ROPE" and self.position_embedding_type != "ALIBI":
            error_msg = "`pe_type` is only support for type: `ROPE` and `ALIBI`, loaded from config.json -> pe_type."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(error_msg)
        self.skip_word_embedding = config.skip_word_embedding

    @property
    def name(self):
        return "llama"

    def get_in_tensor_names(self):
        default_input = ['input_ids', 'position_ids', 'token_offset', 'seq_len', 'layer_id']
        if self.config.pe_type == "ROPE":
            default_input.extend(['cos_table', 'sin_table'])
        default_input.extend(['attention_mask', 'lm_head_indices'])
        return default_input

    def get_out_tensor_names(self):
        return ['model_out']

    def build_graph(self, graph, is_prefill):
        # 设置输入输出
        kv_cache_names = []
        for i in range(self.config.num_hidden_layers):
            kv_cache_names.extend([f"layer_{i}_k_cache", f"layer_{i}_v_cache"])
        graph.add_input_output(
            input=list(self.weight.keys()) + kv_cache_names + self.get_in_tensor_names(),
            output=self.get_out_tensor_names())

        # 增加图节点
        self.model.build_graph(graph, is_prefill)

        lm_head_linear_param = {
            "op_name": "lm_head_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.lm_head.linear,
            "default_dtype": self.dtype,
        }
        lm_head_linear_parallel_param = {
            "op_name": "lm_head_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_GATHER,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend),
            "unpad_inputs": False,
            "linear_param": lm_head_linear_param,
        }
        lm_head_param = {
            "op_name": "test_lm_head",
            "category": CommonOpBuilderType.LM_HEAD,
            "enable_linear_parallel": True,
            "linear_parallel_param": lm_head_linear_parallel_param,
            "gather_ahead": True if is_prefill else False,
            "unpad_inputs": False
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

    def prepare_inputs(self, input_ids_or_embedding: torch.Tensor, position_ids: torch.Tensor,
                        cu_seqlen_prefill: bool | None, max_seq_len: int):
        target_key = PREFILL if cu_seqlen_prefill else DECODE
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
        token_offset = torch.tensor([int(self.token_offset[0])] * self.batch_num, dtype=torch.int32, device=self.device)
        self.graph_inputs[target_key].update({
            "input_ids": input_ids_or_embedding,
            "position_ids": position_ids.to(torch.int64),
            "seq_len": self.seq_len_encoder if cu_seqlen_prefill else self.seq_len_decoder,
            "cos_table": self.cos_embed,
            "sin_table": self.sin_embed,
            "token_offset": token_offset,
        })
        if cu_seqlen_prefill:
            self.graph_inputs[target_key]["lm_head_indices"] = torch.tensor(
                [self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.device)
        else:
            self.graph_inputs[target_key]["lm_head_indices"] = self.lm_head_indices_fake
        
        real_vocab_size = self.weight.get(f"{self.lm_head_prefix}.weight").shape[0] * self.tp_world_size
        self.graph_outputs[target_key][self.get_out_tensor_names()[0]] = \
            torch.ones(self.batch_num, 1, real_vocab_size, dtype=self.dtype, device=self.device)
        self.graph_param[target_key]['token_offset'] = token_offset.cpu()
        self.graph_param[target_key]['seq_len'] = self.seq_len_encoder.cpu() if cu_seqlen_prefill \
            else self.seq_len_decoder.cpu()
