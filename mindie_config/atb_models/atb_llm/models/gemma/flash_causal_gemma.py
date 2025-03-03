# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from typing import Optional, List, Tuple
import torch
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from .config_gemma import GemmaConfig
from .modeling_gemma import FlashGemmaModel
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import AttnWrapper, MlpWrapper
from ...utils.layers import load_column_multi, TensorHead
from ...utils.dist import get_rank_table_file


class FlashGemmaForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.model = FlashGemmaModel(config, weights)
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="model.embed_tokens",
                weights=weights,
                is_norm=False,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=["model.embed_tokens"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.config = config
        self.head_dim = config.head_dim
        self.in_tensor_length = 12
        self.total_head_nums = config.hidden_size // self.head_dim
        self.acl_encoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.position_embedding_type = config.pe_type
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.atten_mask_cpu = None
        self.skip_word_embedding = False
        if self.position_embedding_type != "ROPE" and self.position_embedding_type != "ALIBI":
            logger.error("error: only support petype: ROPE and ALIBI, check your config.json: pe_type",
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise AssertionError(f'petype: {self.position_embedding_type} not supported')
        
        self.cos_embed = None
        self.sin_embed = None
        self.ascend_weight = None
        self.acl_param = None
        

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: GemmaConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("gemma_GemmaDecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("gemma_GemmaDecoderModel")
    
    def get_weights(self):
        weight_wrapper = self.get_weight_wrapper()
        return weight_wrapper

    def init_ascend_weight(self):
        encoder_param, decoder_param = self.get_coder_param()
        encoder_param["hiddenSize"] = self.hidden_size
        decoder_param["hiddenSize"] = self.hidden_size
        encoder_param["skipWordEmbedding"] = False
        decoder_param["skipWordEmbedding"] = False
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        if self.rope_given_inv_feq_str is None and self.rope_vanilla_theta is None:
            self._init_rope_cos_sin(max_seq_len, dtype, device)
        else:
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len, dim, dtype, device, 0, self.rope_mscale,
                self.rope_keep_local_base_windows, self.rope_theta,
                self.rope_vanilla_theta, self.rope_given_inv_feq_str
            )

    def prepare_inputs_for_ascend(
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
        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            self.init_cos_sin_table(self.max_position_embeddings, self.head_dim, self.dtype, self.device)

            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = self.cos_embed
            self.acl_encoder_operation_inputs[3] = self.sin_embed
            self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = self.placeholder
            self.acl_encoder_operation_inputs[8] = self.placeholder
            self.acl_encoder_operation_inputs[9] = self.placeholder
            self.acl_encoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[11] = lm_head_indices.to(torch.int64)
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            atten_mask = self.attn_mask_fake
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.cos_embed
            self.acl_decoder_operation_inputs[3] = self.sin_embed
            self.acl_decoder_operation_inputs[4] = atten_mask
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = self.placeholder
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[11] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param
    
    # 固定基频: rope_theta
    # 自定义基频: rope_given_inv_feq_str
    # 分段基频: rope_theta/rope_given_inv_feq_str + rope_vanilla_theta + rope_keep_local_base_windows
    def _get_cos_sin_table(self, max_seq_len, dim, dtype, device, offset=0, mscale=1,
                        keep_local_base_windows=None, rope_theta=None, rope_vanilla_theta=None, given_inv_feq_str=None):
        if given_inv_feq_str:
            inv_freq = torch.FloatTensor([float(invf) for invf in given_inv_feq_str.split(',')], device=device)
            if len(inv_freq) != dim // 2:
                logger.error("error: only support len(inv_freq) == dim/2 , check your inv_freq length",
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
                    ErrorCode.ATB_MODELS_EXECUTION_FAILURE
                )
                raise AssertionError('keep_local_base_windows: length not match head_dim/2')

            inv_freq_base = 1.0 / (rope_vanilla_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
            freqs_base = torch.outer(seq, inv_freq_base)
            freqs_after_window = freqs + torch.tensor(keep_local_base_windows) * (inv_freq_base - inv_freq)
            for idx, i_keep_local_base_window in enumerate(keep_local_base_windows):
                freqs[:, idx] = torch.cat((freqs_base[:i_keep_local_base_window, idx], 
                freqs_after_window[i_keep_local_base_window:, idx]))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation（ks）
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(dtype).to(device), (emb.sin() * mscale).to(dtype).to(device)

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        self.rotary_embedding.update_cos_sin_cache_total(dtype, device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
        