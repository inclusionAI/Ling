# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
import json
import math
from typing import List, Optional, Tuple

import torch
from atb_llm.utils.log import logger
from atb_llm.utils.layers import load_column_multi

from .modeling_bloom import FlashBloomModel
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ..base.flash_causal_lm import FlashForCausalLM
from .config_bloom import BloomConfig
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.env import ENV


CPP_BLOOM_MODEL_CLASS_NAME = "bloom_BloomDecoderModel"


def _get_interleave(n, alibi_bias_max=8.0):
    def _get_interleave_power_of_2(n, alibi_bias_max):
        if n == 0:
            return 0
        else:
            start = (0.5 ** (alibi_bias_max / n))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n, alibi_bias_max)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
            _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _gen_alibi_mask(n_head, max_pos, alibi_bias_max=8.0):
    slopes = torch.Tensor(_get_interleave(n_head, alibi_bias_max))
    alibi = torch.arange(max_pos)
    alibi = (alibi[None, :] - alibi[:, None]).expand(n_head, -1, -1)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


class FlashBloomForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if not hasattr(config, 'model_max_length'):
            config.model_max_length = 4096  # default model_max_length=4096
        if not hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = config.model_max_length
        if not hasattr(config, 'alibi_bias_max'):
            config.alibi_bias_max = 8.0  # default alibi_bias_max=8.0
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights, **kwargs)
        del self.rotary_embedding
        self.model = FlashBloomModel(config, weights)
        self.lm_head_weight = None  # lmhead share weights with embedding
        extra_prefix = "" if not config.quantize else "transformer."  # msmodelslim 量化后的模型会多一个 transformer 的前缀
        self.lm_head = load_column_multi(
                config,
                prefixes=[f"{extra_prefix}word_embeddings"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        self.config = config  # for quantize
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        # for alibi
        self.training = False
        self.first_run = True
        self.max_cache_pos = config.model_max_length
        self.n_head = config.num_attention_heads  # total head num

        # trans data
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)
        self.init_ascend_weight_flag = False

        self.in_tensor_length = 12
        self.acl_operation_inputs = []

        self.alibi_bias_max = config.alibi_bias_max
        self.atten_mask_cpu = None
        self.alibi_mask_compress = False
        self.ascend_weight = None
        self.weight_linear_type = None
        self.weight_pack_quant_type = None
        self.weight_linear_transpose_types = None

    def init_ascend_weight(self):
        logger.info(f">>>> quant-{self.quantize}")
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        self.weight_linear_type = weight_wrapper.linear_type
        self.weight_pack_quant_type = weight_wrapper.pack_quant_type
        self.weight_linear_transpose_types = weight_wrapper.linear_transpose_types
        mm_cnt = 0
        for t in range(1, len(self.ascend_weight)):
            # 如果不是 W8A16 的 bias, 就不需要处理
            if self.ascend_weight[t - 1].dtype != torch.int8:
                continue
            mm_cnt += 1
            if mm_cnt % 2 == 0:
                self.ascend_weight[t] /= max(1., self.tp_world_size)
        rank_table_file = ENV.rank_table_file
        position_embedding_type = PositionEmbeddingType.ALIBI
        _common_param = {
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": False,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "enableSwiGLU": False,
            "normType": NormType.LAYER_NORM,
            "normEps": self.config.layer_norm_epsilon,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "positionEmbeddingType": position_embedding_type,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": self.weight_pack_quant_type,
            "linearQuantType": self.weight_linear_type,
            "linearTransposeType": self.weight_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "quantGroupSize": self.config.quantization_config.group_size,
            "rankTableFile": rank_table_file,
            "linearHasBias": [[True, True, True, True]] * self.config.num_hidden_layers
        }
        self.acl_encoder_operation.set_param(json.dumps({**_common_param, "isPrefill": True}))
        self.acl_decoder_operation.set_param(json.dumps({**_common_param, "isPrefill": False}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_ascend_operations(self, config: BloomConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_BLOOM_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_BLOOM_MODEL_CLASS_NAME)

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.first_run:
            self.first_run = False
            self.register_buffer(
                "future_mask",
                _gen_alibi_mask(self.n_head, self.max_cache_pos).to(tensor),
                persistent=False,
            )
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.register_buffer(
                "future_mask",
                _gen_alibi_mask(self.n_head, self.max_cache_pos).to(tensor),
                persistent=False,
            )
        mask = self.future_mask[: self.n_head, :seq_length_with_past, :seq_length_with_past]
        if self.tp_world_size > 1:
            mask = mask.chunk(self.tp_world_size, dim=0)
            mask = mask[self.tp_rank]
        return mask

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
        max_seq_len = self.max_cache_pos
        q_lens = kwargs.get('q_lens', [])
        if is_prefill and lm_head_indices is None:  # prefill
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        atten_mask = self.get_alibi_mask(self.place_holder, max_seq_len)
        if not is_prefill:
            atten_mask = atten_mask[:, -1:, :]

        if self.soc_info.need_nz:
            atten_mask = self.transdata_operation.execute([atten_mask])[0]
        self.acl_operation_inputs = [
            input_ids,
            self.place_holder,
            self.place_holder,
            self.place_holder,
            atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            input_lengths.to(torch.int32),
            input_lengths.to(torch.int32),
            self.place_holder,
            input_lengths.to(torch.int32),
            lm_head_indices if is_prefill else self.lm_head_indices_fake
        ]
        acl_param = json.dumps({"seqLen": input_lengths.tolist(), "qLen": q_lens})
        return self.acl_operation_inputs, acl_param

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attention',
            pack_name='query_key_value',
            sep_names=None,
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='dense_h_to_4h',
            sep_names=None,
            down_name='dense_4h_to_h'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.word_embeddings)
        weight_wrapper.register_norm(self.model.word_embeddings_layernorm)
        for i in range(self.config.num_hidden_layers):
            layer = self.model.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attention
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper