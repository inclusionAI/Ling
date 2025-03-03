# Copyright (c) 2023, Baichuan Intelligent Technology. All rights reserved.
import json
import math
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch_npu
from atb_llm.models.baichuan.v2_13b.config_baichuan import BaichuanConfig
from atb_llm.utils.env import ENV
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from torch import nn

from ..modeling_baichuan import FlashBaichuanModel
from ...base.flash_causal_lm import FlashForCausalLM
from ....utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
                _get_interleave_power_of_2(closest_power_of_2)
                + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _gen_alibi_mask(n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


class FlashBaichuanForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if not hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = config.model_max_length
        super().__init__(config, weights, **kwargs)
        del self.rotary_embedding
        self.model = FlashBaichuanModel(config, weights)
        self.lm_head_weight = None
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
            norm=config.vocab_size == 125696

        )
        self.config = config  # for quantize
        if self.dtype != torch.float16:
            error_msg = f"Unsupported type: {self.dtype}. " \
                        f"Only the `float16` type is supported. Modify the `torch_dtype` field in the config.json."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(error_msg)
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')
        # for intensor input
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []

        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None

        # for alibi
        self.first_run = True
        self.is_alibi_mask_free = ENV.ailbi_mask_enable
        self.max_cache_pos = config.model_max_length
        self.n_head = config.num_attention_heads
        self.alibi_base_len = 256
        self.first_gen_slope = True
        self.slopes = None
        self.ascend_weight = None
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None


        # trans data
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)

    def init_ascend_weight(self):
        logger.info(f">>>> quant-{self.quantize}")
        self.ascend_weight, linear_types, pack_quant_configs, linear_transpose_types = self.get_weights()
        coder_param = {
            "isFA": False,
            "isUnpadInputs": True,
            "isBF16": False,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "enableSwiGLU": not self.soc_info.need_nz,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableAlibiMaskFree": not self.soc_info.need_nz and self.is_alibi_mask_free,
            "enableCompressHead": self.compress_head_enable,
            "enableAddNorm": False,
            "positionEmbeddingType": PositionEmbeddingType.ALIBI,
            "quantGroupSize": self.config.quantization_config.group_size
        }
        encoder_param = {**coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable}
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        logger.info(">>>> baichuan2_13b_PagedAttentionParam is inited.")
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_ascend_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            logger.debug(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.debug(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def init_ascend_operations(self, config: BaichuanConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_PagedAttentionQuantModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_PagedAttentionQuantModel")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        head_nums = self.config.num_attention_heads // self.tp_world_size
        batch_size = input_lengths.shape[0]
        block_size = kv_cache[0][0].shape[1]
        input_lengths_js = input_lengths
        if is_prefill:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            if self.compress_head_enable:
                wins = [
                    105, 125, 149, 178, 211, 251, 299, 356, 423, 503,
                    598, 712, 847, 1007, 1198, 1424, 1694, 2014, 2396, 2849,
                    3388, 4031, 4791, 5699, 6779, 8061, 9583, 11399, 13559, 16117,
                    19176, 22790, 97, 115, 137, 163, 194, 230, 274, 326
                ]
                temp_c = input_lengths.cpu()
                wins_batch = torch.zeros((batch_size, head_nums * self.tp_world_size), dtype=torch.int32).npu()
                for batch_idx in range(batch_size):
                    temp_wins = [min(x, temp_c[batch_idx]) for x in wins]
                    wins_batch[batch_idx][:] = torch.tensor(temp_wins)

                max_out_length = 256
                max_need_blocks = 0
                if block_size != 0:
                    max_need_blocks = math.ceil((max_seq_len + max_out_length) / block_size)
                block_tables = torch.zeros((batch_size, head_nums, max_need_blocks), dtype=torch.int32).npu()

                global_need_blocks = 0
                for batch_idx in range(batch_size):
                    for head_idx in range(head_nums):
                        cur_need_blocks = 0
                        if block_size != 0:
                            cur_need_blocks = math.ceil((wins_batch[batch_idx][head_idx + self.tp_rank * head_nums] \
                                                         + max_out_length) / block_size)
                        block_tables[batch_idx][head_idx][0:cur_need_blocks] = torch.arange(0, cur_need_blocks) \
                                                                               + global_need_blocks
                        global_need_blocks = global_need_blocks + cur_need_blocks

                slots = torch.zeros((batch_size, head_nums), dtype=torch.int)
                self.decoder_slots = torch.zeros((batch_size, head_nums), dtype=torch.int)
                for batch_idx in range(batch_size):
                    for head_idx in range(head_nums):
                        offset = int(block_tables[batch_idx][head_idx][0].cpu().item()) * block_size
                        slots[batch_idx][head_idx] = offset
                        seq_len = wins_batch[batch_idx][head_idx + self.tp_rank * head_nums]
                        self.decoder_slots[batch_idx][head_idx] = slots[batch_idx][head_idx] + seq_len - 1

                slots = slots.npu()
                self.decoder_slots = self.decoder_slots.npu()

                block_tables = block_tables.reshape(batch_size * head_nums, max_need_blocks)
                slots = slots.reshape(batch_size * head_nums)
                self.decoder_slots = self.decoder_slots.reshape(batch_size * head_nums)
                self.all_wins_batch = wins_batch[:, self.tp_rank * head_nums: (self.tp_rank + 1) * head_nums]
                self.all_wins_batch = self.all_wins_batch.reshape(-1)
                wins_batch = wins_batch.reshape(-1)

                self.block_tables_global = block_tables
                self.wins_global = wins_batch
                self.wins_batch_1 = torch.ones((batch_size, head_nums * self.tp_world_size), dtype=torch.int32).npu()
        else:
            if self.compress_head_enable:
                self.decoder_slots = self.decoder_slots + 1
                slots = self.decoder_slots
                block_tables = self.block_tables_global
                self.wins_global = self.wins_global + 1
                self.wins_global = self.wins_global.reshape(batch_size, -1)
                input_lengths_js = self.wins_global[:, self.tp_rank * head_nums: (self.tp_rank + 1) * head_nums]
                input_lengths_js = input_lengths_js.reshape(-1).cpu()

                self.all_wins_batch = self.wins_batch_1[:, self.tp_rank * head_nums: (self.tp_rank + 1) * head_nums]
                self.all_wins_batch = self.all_wins_batch.reshape(-1)
                input_lengths[:] = 1

        acl_param = json.dumps({
            "seqLen": input_lengths_js.tolist() if self.compress_head_enable else input_lengths.tolist(),
        })

        # generate attention_mask
        attention_mask = self.generate_mask(max_seq_len, is_prefill)

        if not self.soc_info.need_nz and self.is_alibi_mask_free:
            self.slopes = self.get_slopes()
            if self.tp_world_size < 1:
                error_msg = "The tp_world_size should larger than 0."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not is_prefill:
                attention_mask = attention_mask * \
                                 self.slopes.view(int(self.n_head / self.tp_world_size), 1, 1).to(torch.float16)
        else:
            self.slopes = self.place_holder

        if is_prefill:
            self.acl_encoder_operation_inputs = [
                input_ids,  # 0
                self.place_holder,
                self.place_holder,
                self.place_holder,
                attention_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.place_holder,
                self.place_holder,
                self.place_holder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64)  # 11
            ]
            if self.compress_head_enable:
                self.acl_encoder_operation_inputs.append(self.all_wins_batch.to(torch.int32))
                self.acl_encoder_operation_inputs.append(input_lengths_js.npu().to(torch.int32))
            self.acl_encoder_operation_inputs.append(self.slopes.to(torch.float))
            return self.acl_encoder_operation_inputs, acl_param
        else:
            self.acl_decoder_operation_inputs = [
                input_ids,
                self.place_holder,
                self.place_holder,
                self.place_holder,
                attention_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.place_holder,
                self.place_holder,
                self.place_holder,
                input_lengths.to(torch.int32),
                self.lm_head_indices_fake  # 11
            ]
            if self.compress_head_enable:
                self.acl_decoder_operation_inputs.append(self.all_wins_batch.to(torch.int32))
                self.acl_decoder_operation_inputs.append(input_lengths_js.npu().to(torch.int32))
            self.acl_decoder_operation_inputs.append(self.slopes.to(torch.float))
            return self.acl_decoder_operation_inputs, acl_param

    def ntoken_transdata(self, tensor):
        """
        prefill: [batch , head_num,max_s,max_s] -> [batch * head_num, maxS/16, maxS, 16]
        prefill: [4, 40, 1024, 1024]  ->  [160, 64, 1024, 16]
        max_s不够16整除的要pad 如[4,40,17,17] -> [4, 40, 17, 32] -> [160,2,17,16]

        decode: [batch,head_num,1,max_s] -> [batch * head_num, max_s/16, 16, 16]
        max_s不够16整除的要pad 如[1,40,1,17] -> [1, 40, 1, 32] -> [1, 40, 16, 32] ->[40,2,16,16]
        """
        return self.transdata_operation.execute([tensor])[0]

    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='W_pack',
            sep_names=None,
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.config.num_hidden_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return (weight_wrapper.weights, weight_wrapper.linear_type,
                weight_wrapper.pack_quant_type, weight_wrapper.linear_transpose_types)

    def get_slopes(self):
        if self.first_gen_slope:
            self.first_gen_slope = False
            slopes_value = torch.Tensor(_get_interleave(self.n_head))
            if self.tp_world_size < 1:
                error_msg = "The tp_world_size should larger than 0."
                logger.error(error_msg)
                raise ValueError(error_msg)
            split_nums = int(self.n_head / self.tp_world_size)
            slopes = torch.split(slopes_value, split_nums)[self.tp_rank].to(torch.float).to('npu')
            self.register_buffer("slopes_gen", slopes, persistent=False)
        return self.slopes_gen

    def get_free_first_alibi(self, tensor):
        max_pos = self.alibi_base_len
        position_point = torch.arange(max_pos) - max_pos + 1
        position_point = position_point.unsqueeze(0).unsqueeze(0).expand(self.n_head, -1, -1)
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        alibi = position_point.view(self.n_head, 1, max_pos)
        alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
        alibi_mask = alibi_mask.unsqueeze(0) + alibi
        attention_mask = alibi_mask.to(tensor)
        return attention_mask

    def get_free_no_first_alibi(self, seq_length_with_past):
        if self.first_run:
            self.first_run = False
            self.register_buffer(
                "free_future_mask",
                torch.arange(self.max_cache_pos).unsqueeze(0).unsqueeze(0).expand(self.n_head, 1, -1),
                persistent=False,
            ),
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.register_buffer(
                "free_future_mask",
                torch.arange(self.max_cache_pos).unsqueeze(0).unsqueeze(0).expand(self.n_head, 1, -1),
                persistent=False,
            ),
        mask = self.free_future_mask[:, :, :seq_length_with_past]
        if self.tp_world_size > 1:
            mask = mask.chunk(self.tp_world_size, dim=0)
            mask = mask[self.tp_rank]
        return mask

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.first_run:
            self.first_run = False
            self.register_buffer(
                "future_mask",
                _gen_alibi_mask(self.n_head, self.max_cache_pos).to(
                    tensor
                ),
                persistent=False,
            )
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.register_buffer(
                "future_mask",
                _gen_alibi_mask(self.n_head, self.max_cache_pos).to(
                    tensor
                ),
                persistent=False,
            )
        mask = self.future_mask[: self.n_head, :seq_length_with_past, :seq_length_with_past]
        if self.tp_world_size > 1:
            mask = mask.chunk(self.tp_world_size, dim=0)
            mask = mask[self.tp_rank]
        return mask

    def generate_mask(self, max_seq_len, is_prefill):
        seq_length_with_past = max_seq_len
        if not self.soc_info.need_nz and self.is_alibi_mask_free:  # 310p is not support alibi free
            if is_prefill:
                attention_mask = self.get_free_first_alibi(self.place_holder)
                attention_mask = attention_mask[0].to(torch.float16).to('npu')
            else:
                attention_mask = self.get_free_no_first_alibi(seq_length_with_past)
                attention_mask = attention_mask.to(torch.float16).to('npu')
        else:
            attention_mask = self.get_alibi_mask(self.place_holder, max_seq_len)
            if not is_prefill:
                attention_mask = attention_mask[:, -1:, :]
            if self.soc_info.need_nz:
                attention_mask = self.ntoken_transdata(attention_mask)
        logger.debug(f"{is_prefill=} and the current alibi mask shape is : {attention_mask.shape}")
        return attention_mask

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  #
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ):
        if self.lm_head_weight is None:
            if self.config.vocab_size == 125696:
                logger.debug("baichuan2 13B normalize lm_head")
                self.lm_head_weight = nn.functional.normalize(self.state_dict()["lm_head.linear.weight"])
            else:
                self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
            if self.soc_info.need_nz:
                self.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head_weight

        # add acl model
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)

        acl_inputs, acl_param = \
            self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                           block_tables, slots, input_lengths, max_seq_len,
                                           lm_head_indices, **kwargs)
        hidden_states = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)

        outputs = tuple(v for v in [hidden_states] if v is not None)
        logits = outputs[0]
        return logits