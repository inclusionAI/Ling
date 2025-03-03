# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional, List, Tuple
from collections import OrderedDict
import torch
from torch import nn
from torch.functional import F
from ..qwen2.flash_causal_qwen2 import FlashQwen2ForCausalLM

MROPE_SECTION = [16, 24, 24]


class TensorEmbeddingWithoutChecking(nn.Module):
    def __init__(self, prefix: str, weights):
        super().__init__()
        weight = weights.get_whole_tensor(f"{prefix}.weight", dim=0)

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(input_tensor, self.weight)
        return out

    def get_weights(self, prefix):
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.weight"] = self.weight.data
        return weight_dict


class FlashQwen2UsingMROPEForCausalLM(FlashQwen2ForCausalLM):
    def __init__(self, config, weights, **kwargs):
        kwargs = {"skip_word_embedding": True, "transformer_wte_parallel": False}
        super().__init__(config, weights, **kwargs)
        model_prefix = kwargs.get("model_prefix", "model")
        self.transformer.wte = TensorEmbeddingWithoutChecking(
            prefix=f"{model_prefix}.embed_tokens", weights=weights
        )
        for p in self.transformer.wte.parameters():
            p.requires_grad = False

    def update_thw_cos_sin(self, position_ids_thw, mrope_section):
        normal_cos = self.rotary_embedding.get_cos_cached_total()
        normal_sin = self.rotary_embedding.get_sin_cached_total()
        cos = normal_cos[position_ids_thw].clone()
        sin = normal_sin[position_ids_thw].clone()
        mrope_section = mrope_section * 2
        cos_thw = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).squeeze(0)
        sin_thw = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).squeeze(0)
        return cos_thw, sin_thw

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

        seqlen = "seqLen"
        acl_param = json.dumps({
            seqlen: input_lengths.tolist()
        })
        cos_table, sin_table = None, None

        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype,
            self.device,
            self.max_position_embeddings
        )
        cos_table = self.rotary_embedding.get_cos_cached_total()
        sin_table = self.rotary_embedding.get_sin_cached_total()

        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(max_seq_len if self.split_fuse_enable else self.max_base_len,
                                                          self.dtype, self.device)
            if self.soc_info.need_nz:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]

        else:
            if self.skip_word_embedding:
                input_ids = self.transformer.wte(input_ids)
            attention_mask = self.attn_mask_fake

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        self.acl_operation_inputs[0] = input_ids
        self.acl_operation_inputs[1] = position_ids
        self.acl_operation_inputs[2] = cos_table
        self.acl_operation_inputs[3] = sin_table
        self.acl_operation_inputs[4] = attention_mask
        self.acl_operation_inputs[5] = block_tables.to(torch.int32)
        self.acl_operation_inputs[6] = slots.to(torch.int32)
        self.acl_operation_inputs[7] = self.placeholder
        self.acl_operation_inputs[8] = self.placeholder
        self.acl_operation_inputs[9] = self.place_holder
        self.acl_operation_inputs[10] = input_lengths.to(torch.int32)
        self.acl_operation_inputs[11] = lm_head_indices

        return self.acl_operation_inputs, acl_param

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()
        self.init_kvcache(kv_cache)

        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)
        position_ids_thw_list = kwargs.pop("position_ids_thw_list", None)
        if is_prefill and position_ids_thw_list is not None and position_ids_thw_list[0] is not None:
            mrope_section = kwargs.pop("mrope_section", MROPE_SECTION)
            cos_list = []
            sin_list = []
            for position_ids_thw in position_ids_thw_list:
                cos_thw, sin_thw = self.update_thw_cos_sin(position_ids_thw, mrope_section)
                cos_list.append(cos_thw)
                sin_list.append(sin_thw)
            acl_inputs[2] = torch.cat(cos_list, dim=0)
            acl_inputs[3] = torch.cat(sin_list, dim=0)
            new_position_ids = torch.arange(input_lengths.sum(), dtype=position_ids.dtype, device=position_ids.device)
            acl_inputs[1] = new_position_ids
            del position_ids_thw_list

        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits
