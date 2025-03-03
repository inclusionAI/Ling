# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from pathlib import Path

import torch

from atb_llm.utils.log import logger
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi, TensorHead
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from .modeling_qwen2 import FlashQwenModel
from .config_qwen2 import Qwen2Config
from ...utils.env import ENV



class FlashQwen2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.transformer = FlashQwenModel(config, weights)
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="lm_head",
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
                        prefixes=["lm_head"],
                        weights=weights,
                        head_size=1,
                        lm_head=True,
                    )
        self.config = config  # for quantize
        self.attn_mask_fake = self.attn_mask.get_attn_mask(1, dtype=self.dtype, device="npu")
        self.place_holder = torch.tensor([1], dtype=self.dtype, device='npu')

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.acl_param = None
        self.acl_operation_inputs = None
        self.ascend_weight = None
        
        # 用于区分是否是gte模型
        self.is_text_embedding = True
        self.is_warmup = True
        self.text_count = 0
        
        # for save embedding path
        self.save_embedding_path = None
        file_path = Path(__file__).resolve().parent
        file_path = str(file_path).split('/')[:-3]
        file_path.extend(["examples", "embedding_tensor", "gte-qwen2"])
        self.save_embedding_path = '/'.join(file_path)
        self.init_ascend_operations(config)    
        
    def init_ascend_operations(self, config: Qwen2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_GteDecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_GteDecoderModel")
        logger.info(f">>>> qwen_DecoderModel is called. \n{config=}")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=None,
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=None,
            down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)               
            if self.soc_info.need_nz:
                del layer.attn
                del layer.ln_2
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        position_embedding_type = PositionEmbeddingType.ROPE
        acl_param_dict = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "withEmbedding": True,
            "isEmbeddingParallel": True,
            "isClassification": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "quantGroupSize": self.config.quantization_config.group_size,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "positionEmbeddingType": position_embedding_type,
            "enableAddNorm": False,
            "rankTableFile": ENV.rank_table_file,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers,
            "isEmbedding": True if self.config.intermediate_size == 18944 else False,
            "enableLogN": False
        }
        acl_param_encoder = json.dumps({**acl_param_dict, "isPrefill": True, "enableLcoc": self.lcoc_enable})
        acl_param_decoder = json.dumps({**acl_param_dict, "isPrefill": False, "enableLcoc": False,
                                        "enableSpeculate":self.speculate_enable})
            
        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)
            
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  **kwargs):
        block_tables = kwargs.get("block_tables", None)
        slots = kwargs.get("slots", None)
        input_lengths = kwargs.get("input_lengths", None)
        lm_head_indices = kwargs.get("lm_head_indices", None)
        self.acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype,
            self.device,
            self.max_position_embeddings
        )
        cos_table = self.rotary_embedding.get_cos_cached_total()
        sin_table = self.rotary_embedding.get_sin_cached_total()
        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            if self.soc_info.need_nz:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        else:
            attention_mask = self.attn_mask_fake
            if self.speculate_enable:
                q_lens = kwargs.get('q_lens', [])
                spec_mask = kwargs.get('spec_mask', None)
                self.acl_param = json.dumps({
                    "seqLen": input_lengths.tolist(),
                    "qLen": q_lens
                })
                q_lens = torch.tensor(q_lens).to(self.device).to(torch.int32)
                req_mask = spec_mask
                if self.soc_info.need_nz:
                    req_mask = self.transdata_operation.execute([req_mask])[0]
                    attention_mask = req_mask
        
        
        self.acl_operation_inputs = [
            input_ids,  # IN_TENSOR_INPUTIDS
            position_ids, # IN_TENSOR_POSITIONIDS
            cos_table,  # IN_TENSOR_COSEMBED
            sin_table,  # IN_TENSOR_SINEMBED
            self.place_holder if self.config.intermediate_size == 18944 else attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            self.place_holder,  # IN_TENSOR_KV_CACHE_IDX
            self.place_holder,  # IN_TENSOR_TOKEN_OFFSET
            self.place_holder,  # IN_HOLDER
            input_lengths.to(torch.int32),  # IN_TENSOR_SEQ_LENGTHS
            lm_head_indices if is_prefill else self.lm_head_indices_fake,  # IN_TENSOR_LOGTIS_INDICES

        ]
        if self.speculate_enable and not is_prefill:
            self.acl_operation_inputs.append(q_lens)
        return self.acl_operation_inputs, self.acl_param
    
    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            **kwargs,
    ) -> torch.Tensor:
        kv_cache = kwargs.get('kv_cache', None)
        block_tables = kwargs.get('block_tables', None)
        slots = kwargs.get('slots', None)
        input_lengths = kwargs.get('input_lengths', None)
        lm_head_indices = kwargs.get('lm_head_indices', None)
        
        if not self.ascend_weight:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(
                                input_ids,
                                position_ids,
                                is_prefill, 
                                block_tables=block_tables, 
                                slots=slots, 
                                input_lengths=input_lengths, 
                                lm_head_indices=lm_head_indices
                            )
        final_norm_out = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        embedding_value = final_norm_out[-1]
        save_path = Path(self.save_embedding_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)  
        if not self.is_warmup:
            logits_name = f"embedding_tensor_{self.text_count}"
            torch.save(embedding_value, f"{self.save_embedding_path}/{logits_name}.pth")
            self.text_count += 1
        self.is_warmup = False
        # when mindIE-Service supports, the return value should be embedding_value for called.
        return final_norm_out