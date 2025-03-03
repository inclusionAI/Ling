# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
from typing import Optional, List, Tuple

import torch

from .modeling_llama_atb import LlamaModelATB
from ..base.flash_causal_lm_atb import FlashForCausalLMATB, PREFILL, DECODE
from ...utils.layers import load_column_multi, TensorHead
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode
from ...common_op_builders.data_type import CommonOpBuilderType
from ...common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from ...common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend


class FlashLlamaForCausalLMATB(FlashForCausalLMATB):
    def __init__(self, config, weights, lm_head_prefix="lm_head", model_prefix="model", **kwargs):
        super().__init__(config, weights, **kwargs)
        # 模型结构相关
        self.backend = CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL
        self.model_prefix = model_prefix
        self.model = LlamaModelATB(config, weights, model_prefix, lm_head_prefix, 
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
            self.lm_head = load_column_multi(
                config,
                prefixes=[lm_head_prefix],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.position_embedding_type = config.pe_type
        self.alibi_bias_max = config.alibi_bias_max
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.atten_mask_cpu = None
        self.alibi_mask_compress = True
        self.skip_word_embedding = False
        if self.position_embedding_type != "ROPE" and self.position_embedding_type != "ALIBI":
            error_msg = "`pe_type` is only support for type: `ROPE` and `ALIBI`, loaded from config.json -> pe_type."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(error_msg)
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None

    @property
    def name(self):
        return "llama"

    def get_in_tensor_names(self, is_prefill):
        default_input = ['input_ids', 'position_ids', 'slots_mapping', 'seq_len']
        if self.config.pe_type == "ROPE":
            default_input.extend(['cos_table', 'sin_table'])
        if is_prefill or self.config.pe_type == "ALIBI":
            default_input.extend(['attention_mask', 'lm_head_indices'])
        if not is_prefill:
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
            "linear_param": lm_head_linear_param,
        }
        lm_head_param = {
            "op_name": "test_lm_head",
            "category": CommonOpBuilderType.LM_HEAD,
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

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        if self.rope_given_inv_feq_str is None and self.rope_vanilla_theta is None:
            self._init_rope_cos_sin(max_seq_len, dtype, device)
        else:
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len, dim, dtype, device, 0, self.rope_mscale,
                self.rope_keep_local_base_windows, self.rope_theta,
                self.rope_vanilla_theta, self.rope_given_inv_feq_str
            )

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
        attention_mask = kwargs.get('attention_mask', None)
        spec_mask = kwargs.get('spec_mask', None)
        if self.position_embedding_type == "ROPE":
            if is_prefill:
                attention_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            elif self.speculate_enable:
                attention_mask = spec_mask
        elif self.position_embedding_type == "ALIBI":
            if is_prefill:
                if self.attention_mask_cpu is None:
                    self.attention_mask_cpu = self._gen_alibi_mask(
                        self.num_attention_heads,
                        self.max_position_embeddings,
                        self.alibi_bias_max)[self.tp_rank * self.num_attention_heads: \
                            (self.tp_rank + 1) * self.num_attention_heads, :, :].to(self.dtype)
                if self.alibi_mask_compress:
                    # 算子要求: 小于128则按实际长度切，大于128则按128切，算子内部扩展到实际长度
                    slice_len = max_seq_len if max_seq_len <= self.max_base_len else self.max_base_len
                    attention_mask = self.attention_mask_cpu[:, :, :slice_len].npu()
                else:
                    attention_mask = self.attention_mask_cpu[:, :max_seq_len, :max_seq_len].npu()
            else:
                attention_mask = self._gen_alibi_mask_decoder(self.num_attention_heads, position_ids.tolist(),
                                                              max_seq_len, self.alibi_bias_max)[:,
                                 self.tp_rank * self.num_attention_heads:(self.tp_rank + 1) * self.num_attention_heads,
                                 :, :].to(self.dtype).npu()
        else:
            error_msg = "`pe_type` is only support for type: `ROPE` and `ALIBI`, loaded from config.json -> pe_type."
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
        if self.soc_info.need_nz and attention_mask is not None:
            attention_mask = self.transdata_operation.execute([attention_mask])[0]
        # cosine & sine embedding
        if is_prefill:
            self.init_cos_sin_table(self.max_position_embeddings, self.head_size, self.dtype, self.device)

        # 更新输入
        target_key = PREFILL if is_prefill else DECODE
        self.graph_inputs[target_key].update({
            "input_ids": input_ids,
            "position_ids": position_ids.to(torch.int64),
            "slots_mapping": slots.to(torch.int32),
            "seq_len": input_lengths.to(torch.int32)
        })
        if attention_mask is not None:  # attention mask
            self.graph_inputs[target_key]["attention_mask"] = attention_mask
        if self.position_embedding_type == "ROPE":  # cosine & sine embedding
            self.graph_inputs[target_key]["cos_table"] = self.cos_embed
            self.graph_inputs[target_key]["sin_table"] = self.sin_embed
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
            torch.empty(batch_size, real_vocab_size, dtype=self.dtype, device=self.device)

        # 准备bind tensor
        self.graph_param[target_key]['seq_len'] = input_lengths.cpu().to(torch.int32)

        if self.speculate_enable and not is_prefill:
            self.graph_param[target_key]['q_len'] = q_len.cpu()

    def _get_interleave(self, n, alibi_bias_max=8.0):
        def _get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (0.5 ** (alibi_bias_max / n))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self._get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def _fill_with_neg_inf(self, t):
        return t.float().fill_(float("-inf")).type_as(t)

    def _gen_alibi_mask(self, n_head, max_pos, alibi_bias_max=8.0):
        slopes = torch.Tensor(self._get_interleave(n_head, alibi_bias_max))
        tensor_list = []
        # 算子要求的压缩alibi mask shape为 [head_num, max_seq, 128]
        for i in range(128):
            tensor = torch.empty(max_pos).fill_(-float('inf'))
            tensor[i:] = -1 * torch.arange(0, max_pos - i)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0).t()
        tensor = tensor.expand(n_head, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        return alibi_mask

    def _gen_alibi_mask_decoder(self, n_head, pos_list, max_pos, alibi_bias_max=8.0):
        slopes = torch.Tensor(self._get_interleave(n_head, alibi_bias_max))
        tensor_list = []
        for pos in pos_list:
            tensor = torch.empty(max_pos).fill_(-float('inf'))
            tensor[:pos + 1] = torch.arange(-pos, 1)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0)
        tensor = tensor.expand(n_head, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        return alibi_mask.permute(1, 0, 2).unsqueeze(2)

    # 固定基频: rope_theta
    # 自定义基频: rope_given_inv_feq_str
    # 分段基频: rope_theta/rope_given_inv_feq_str + rope_vanilla_theta + rope_keep_local_base_windows
    def _get_cos_sin_table(self, max_seq_len, dim, dtype, device, offset=0, mscale=1,
                           keep_local_base_windows=None, rope_theta=None, rope_vanilla_theta=None,
                           given_inv_feq_str=None):

        if given_inv_feq_str:
            inv_freq = torch.FloatTensor([float(invf) for invf in given_inv_feq_str.split(',')], device=device)
            if len(inv_freq) != dim // 2:
                logger.error("error: only support len(inv_freq) == dim/2 ,check your inv_freq length", 
                             ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise AssertionError('given_inv_feq_str: length not match head_dim/2')
        else:
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

        seq = torch.arange(max_seq_len, device=device).float() + offset
        freqs = torch.outer(seq, inv_freq)

        if keep_local_base_windows:
            keep_local_base_windows = [int(w) for w in keep_local_base_windows.split(',')]
            if len(keep_local_base_windows) != dim // 2:
                logger.error(
                    "error: only support len(keep_local_base_windows) == dim/2 , check your base_windows length", 
                    ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise AssertionError('keep_local_base_windows: length not match head_dim/2')

            inv_freq_base = 1.0 / (rope_vanilla_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
            freqs_base = torch.outer(seq, inv_freq_base)
            freqs_after_window = freqs + torch.tensor(keep_local_base_windows) * (inv_freq_base - inv_freq)
            for idx, i_keep_local_base_window in enumerate(keep_local_base_windows):
                freqs[:, idx] = torch.cat((
                    freqs_base[:i_keep_local_base_window, idx],
                    freqs_after_window[i_keep_local_base_window:, idx]
                ))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation（ks）
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(dtype).to(device), (emb.sin() * mscale).to(dtype).to(device)

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        if self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                             device,
                                                             max_seq_len)

        else:
            scaling_type = self.config.rope_scaling.rope_type
            if scaling_type is None:
                scaling_type = self.config.rope_scaling.type
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                                 device,
                                                                 max_seq_len)
            elif scaling_type == "dynamic":
                logger.error("error: only support scaling type: linear, check your config.json: scaling type", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                raise ValueError(f"not support RoPE scaling type {scaling_type}")
            else:
                logger.error("error: only support scaling type: linear, check your config.json: scaling type", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
