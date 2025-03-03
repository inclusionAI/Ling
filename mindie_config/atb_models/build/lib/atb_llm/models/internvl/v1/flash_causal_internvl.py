# Copyright (c) 2023 OpenGVLab
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import math
import warnings
from typing import Optional, List, Tuple

import torch
from torch import nn
import numpy as np

from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.internvl.modeling_intern_vit import InternVisionModel
from atb_llm.models.internvl.data_preprocess_internvl import (
    load_video, load_and_preprocess_image, create_standardization_params,
    internvl_tensor_parallel_split, IMAGENET_MEAN, IMAGENET_STD
)
from atb_llm.models.internvl.input_builder_internvl import INTERNVL_SYSTEM_PROMPTS
from atb_llm.models.llama.flash_causal_llama import FlashLlamaForCausalLM
from atb_llm.models.llama.config_llama import LlamaConfig
from atb_llm.utils.env import ENV
from atb_llm.utils import shm_utils
from atb_llm.utils.dist import get_rank_table_file
from atb_llm.utils.dist import initialize_torch_distributed
from atb_llm.utils.log.logging import logger, print_log
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.linear.linear import ColumnLinear, RowLinear
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionRotaryEmbedding
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.layers import TensorEmbedding

# Preprocessing params
RESCALE_FACTOR = 1 / 255
CONV_CHANNELS = 3
CONV_GROUPS = 3
IMAGE_SIZE = 448
MAX_NUM_PATCHES = 12

CPP_LLAMA_MODEL_CLASS_NAME = "llama_LlamaDecoderModel"
INTERNVL2_LLAMA3_76B_VOCAB_SIZE = 128265


class FlashInternvlForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.cos_embed = None
        self.sin_embed = None
        self.ascend_weight = None
        self.acl_param = None
        self.acl_operation_inputs = None
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights, **kwargs)
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.quantize = None
        self.skip_word_embedding = True
        self.weights = weights
        self.config = config
        self.vision_config = config.vision_config
        self.llm_config = config.llm_config
        self.llm_config.quantize = None
        self.config.eos_token_id = self.llm_config.eos_token_id

        image_size = config.force_image_size or self.vision_config.image_size
        patch_size = self.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.neftune_alpha = None
        self.num_layers = self.llm_config.num_hidden_layers
        self.rms_norm_eps = self.llm_config.rms_norm_eps
        self.num_attention_heads = self.llm_config.num_attention_heads  # 48
        self.num_hidden_layers = self.llm_config.num_hidden_layers  # 48
        self.num_key_value_heads = self.llm_config.num_key_value_heads  # 8
        self.vocab_size = self.llm_config.vocab_size
        self.hidden_size = self.llm_config.hidden_size  # 6144
        self.head_size = self.hidden_size // self.num_attention_heads  # 128
        # if num_key_value_heads is nondivisible
        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
                                   // self.tp_world_size
        self.npu_id = weights.device.index
        self.template = config.template
        self.ps_version = config.ps_version
        if self.template not in ['Hermes-2', 'internlm2-chat', 'phi3-chat', 'internvl2_5']:
            raise ValueError(
                f"Unsupported template {self.template}, supported templates are `Hermes-2`, "
                "`internlm2-chat`, `phi3-chat`, `internvl2_5`. Please check the value of 'template' in config.json"
            )
        if self.ps_version not in ['v1', 'v2']:
            raise ValueError(
                f"Unsupported ps_version {self.ps_version}, supported templates are `v1` and `v2`."
                "Please check the value of 'ps_version' in config.json"
            )
        self.process_group, self.device = initialize_torch_distributed(self.tp_rank, self.npu_id, self.tp_world_size)

        self.language_model = FlashLlamaForCausalLM(self.llm_config,
                                                    self.weights,
                                                    "language_model.lm_head",
                                                    "language_model.model")
        if self.vocab_size == INTERNVL2_LLAMA3_76B_VOCAB_SIZE:
            self.language_model.model.embed_tokens = TensorEmbedding(
                prefix="language_model.model.embed_tokens", weights=weights
            )

        self.vision_model = InternVisionModel(self.vision_config, self.process_group)
        self.init_module_weight(self.vision_model, weights, prefix="vision_model")
        self.vision_model.to(
            device=weights.device, dtype=weights.dtype
        )

        self.downsample_ratio = config.downsample_ratio
        vit_hidden_size = self.vision_config.hidden_size
        llm_hidden_size = self.llm_config.hidden_size
        if self.downsample_ratio == 0:
            raise ZeroDivisionError("Downsample ratio is zero")
        input_dim = vit_hidden_size * int(np.divide(1, self.downsample_ratio)) ** 2

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            ColumnLinear(input_dim, llm_hidden_size, gather_output=False, process_group=self.process_group),
            nn.GELU(),
            RowLinear(llm_hidden_size, llm_hidden_size, process_group=self.process_group)
        )
        self.init_module_weight(self.mlp1, weights, prefix="mlp1")
        self.mlp1.to(
            device=weights.device, dtype=weights.dtype
        )

        self.dim = self.head_size
        self.base = self.llm_config.rope_theta
        self.scaling_factor = 1.0
        self.max_position_embeddings = self.llm_config.max_position_embeddings
        self.rope_scaling = self.llm_config.rope_scaling
        self.max_seq_len_cached = self.max_position_embeddings
        self.rotary_embedding_device = 'cpu'

        if self.rope_scaling is None:
            print_log(self.tp_rank, logger.info, 'now \033[33m scaling_type: base rope \033[0m')
            self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                                   device=self.rotary_embedding_device,
                                                                   scaling_factor=self.scaling_factor).to(self.device)
        else:
            self.scaling_type = self.rope_scaling.type
            if self.scaling_type == "linear":
                print_log(self.tp_rank, logger.info, f'now \033[33m scaling_type: {self.scaling_type} \033[0m')
                self.scaling_factor = self.rope_scaling.factor  # t=t/scaling_factor
                self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                                       device=self.rotary_embedding_device,
                                                                       scaling_factor=self.scaling_factor).to(
                    self.device)
            elif self.scaling_type == "dynamic":
                print_log(self.tp_rank, logger.info, f'now \033[33m scaling_type: {self.scaling_type} \033[0m')
                self.rope_scaling_factor = self.rope_scaling.factor  # Dynamic NTK 外推方法的系数
                self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                                       device=self.rotary_embedding_device,
                                                                       scaling_factor=self.scaling_factor,
                                                                       ).to(self.device)
            else:
                print_log(self.tp_rank, logger.info, f'now \033[33m scaling_type: {self.scaling_type} \033[0m')
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")

        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(
                f"Unsupported dtype: {self.dtype}, supported dtype are `float16` and `bfloat16`."
                "Please check the value of 'torch_dtype' in config.json"
            )
        
        if self.llm_config.num_attention_heads == 0:
            raise ZeroDivisionError("LLM config num_attention_heads is zero")
        self.head_dim = np.divide(self.llm_config.hidden_size, self.llm_config.num_attention_heads)
        self.in_tensor_length = 13
        self.acl_encoder_operation_inputs = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs = [None] * self.in_tensor_length

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.processor = safe_get_tokenizer_from_pretrained(self.config.model_name_or_path, 
                                                            trust_remote_code=self.trust_remote_code,
                                                            use_fast=False)
        self.img_begin_id = self.processor.encode("<img>")[-1]
        self.img_end_id = self.processor.encode("</img>")[-1]
        self.img_context_token_id = self.processor.encode("<IMG_CONTEXT>")[-1]

        self.init_normalizer()
    
    def init_normalizer(self):
        weight, bias = create_standardization_params(IMAGENET_MEAN, IMAGENET_STD, RESCALE_FACTOR, CONV_CHANNELS)
        self.normalizer = nn.Conv2d(in_channels=CONV_CHANNELS, out_channels=CONV_CHANNELS, kernel_size=1, \
            groups=CONV_GROUPS)
        self.normalizer.weight = nn.Parameter(data=weight, requires_grad=False)
        self.normalizer.bias = nn.Parameter(data=bias, requires_grad=False)
        self.normalizer.npu()
        # Normalizer warmup
        self.normalizer(torch.randn(MAX_NUM_PATCHES, CONV_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device='npu'))

    def init_module_weight(self, module, weights, prefix="vision_model"):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"{prefix}.{model_weight}"), requires_grad=False
            )
            saved_weight = internvl_tensor_parallel_split(model_weight, prefix, \
                                                          self.tp_rank, self.tp_world_size, saved_weight)
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.max_seq_len_cached = self.max_position_embeddings
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: LlamaConfig):
        logger.info("using Llama_DecoderModel")
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_LLAMA_MODEL_CLASS_NAME)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
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
        weight_wrapper.register_embedding(self.language_model.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.language_model.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.language_model.model.norm)
        weight_wrapper.register_model_lmhead(self.language_model.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        rank_table_file = get_rank_table_file()
        # 设置模型参数
        coder_param = {
            "normEps": self.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.vocab_size > 200000,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.language_model.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "enableReduceQuant": self.config.quantization_config.reduce_quant_type is not None,
            "attnBackend": self.attn_decode_backend,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "splitWithStride": False,
            "hiddenSize": self.hidden_size,
            "gemma": False,
            "enableAddNorm": False,
            "enableCompressHead": False,
            "enableLora": self.adapter_manager is not None,
            "quantGroupSize": self.config.quantization_config.group_size,
            "isLongSeq": ENV.long_seq_enable,
        }
        encoder_param = {
            **coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable,
            "skipWordEmbedding": self.skip_word_embedding,
            "enableSpeculate": False,
            "enableSplitFuse": self.split_fuse_enable
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False,
            "enableSpeculate": self.speculate_enable,
            "enablePrefixCache": self.prefix_cache_enable
        }

        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        if scale_factor == 0:
            raise ZeroDivisionError("Scale factor is zero")
        x = x.view(n, w, int(h * scale_factor), int(np.divide(c, scale_factor)))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        if scale_factor == 0:
            raise ZeroDivisionError("Scale factor is zero")
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(np.divide(c, scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def noised_embed(self, vit_embeds, noise_alpha=5):
        dims = torch.tensor(vit_embeds.size(1) * vit_embeds.size(2))
        if dims == 0:
            raise ZeroDivisionError("Dim of the tensor is zero")
        mag_norm = np.divide(noise_alpha, torch.sqrt(dims))
        noise = torch.zeros_like(vit_embeds).uniform_(-mag_norm, mag_norm)
        return vit_embeds + noise

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def prepare_prefill_token(self, multimodalinputs, processor):
        text = multimodalinputs.text
        image = multimodalinputs.image
        video = multimodalinputs.video
        current_query = ""
        if image is not None:
            use_dynamic_prepro = False if self.ps_version == "v1" else True
            pixel_values = load_and_preprocess_image(image, normalizer=self.normalizer, \
                use_dynamic_prepro=use_dynamic_prepro).to(self.dtype).to(self.device)
            vit_embeds = self.extract_feature(pixel_values).to(self.dtype).to(self.device)
            image_tokens_num = self.num_image_token * vit_embeds.shape[0]
            current_query = (f'<img>{"<IMG_CONTEXT>" * image_tokens_num}</img>\n')
        elif video is not None:
            pixel_values, num_path_list = load_video(video)
            pixel_values = pixel_values.to(self.dtype).to(self.device)
            vit_embeds = self.extract_feature(pixel_values).to(self.dtype).to(self.device)
            current_query = ""
            for i, num_patch in enumerate(num_path_list):
                current_query += (f'Frame{i+1}: '
                    f'<img>{"<IMG_CONTEXT>" * num_patch * self.num_image_token}</img>\n')

        system_prompt = INTERNVL_SYSTEM_PROMPTS[self.ps_version][self.template]
        texts = ('<|im_start|>system\n'
                f'{system_prompt}<|im_end|><|im_start|>user\n')
        texts += current_query
        texts += (f'{text}<|im_end|><|im_start|>assistant\n')

        input_ids = processor.encode(texts)
        input_ids = torch.tensor(input_ids).to(self.device)  # input_ids大于不能转为 bfloat16

        input_embeds = self.language_model.model.embed_tokens(input_ids)

        sequence_length, embedding_size = input_embeds.shape

        input_ids = input_ids.reshape(sequence_length)
        vit_embeds = vit_embeds.reshape(-1, embedding_size)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = input_embeds[selected] * torch.zeros(1, dtype=self.dtype,
                device=self.device) + vit_embeds.reshape(-1, embedding_size)
        except Exception as e:
            raise ValueError(
                f'{e} \ninput_embeds[selected].shape={input_embeds[selected].shape}, '
                f'vit_embeds.shape={vit_embeds.shape}\n'
                f'Please check whether shape of input_embeds[selected] matches the shape of vit_embeds.\n'
                f'If not, please check whether self.img_context_token_id: {self.img_context_token_id} '
                f'and the token-id of "<IMG_CONTEXT>" in processor: {processor.encode("<IMG_CONTEXT>")[-1]} '
                f'are the same'
            ) from e

        input_embeds = input_embeds.reshape(-1, embedding_size)
        return input_embeds

    def prepare_prefill_token_service(self, input_ids):
        input_embeds = self.language_model.model.embed_tokens(input_ids)
        sequence_length, embedding_size = input_embeds.shape
        input_ids = input_ids.reshape(sequence_length)

        if torch.any(torch.eq(input_ids, self.img_begin_id)):
            img_bos_set = torch.where(torch.eq(input_ids, self.img_begin_id))[0].detach().cpu().tolist()
            img_eos_set = torch.where(torch.eq(input_ids, self.img_end_id))[0].detach().cpu().tolist()
            batch_images = []
            batch_size_list = []
            for img_bos, img_eos in zip(img_bos_set, img_eos_set):
                if img_eos - img_bos < 2:
                    continue
                image_pixel_value = shm_utils.get_data_from_shm(input_ids[img_bos + 1], input_ids[img_bos + 2],
                    dtype=np.uint8, device=self.device).to(self.dtype)

                batch_images.append(image_pixel_value)
                batch_size_list.append(image_pixel_value.size(0))

            batch_images = torch.cat(batch_images, dim=0)
            batch_images = self.normalizer(batch_images.float()).half()

            vit_embeds = self.extract_feature(batch_images)
            vit_embeds = vit_embeds.to(self.dtype).to(self.device)

            pre_index = 0
            for img_bos, img_eos, batch_size in zip(img_bos_set, img_eos_set, batch_size_list):
                single_vit_embeds = vit_embeds[pre_index: pre_index + batch_size].reshape(-1, embedding_size)
                pre_index += batch_size
                try:
                    input_embeds[img_bos + 1: img_eos] = single_vit_embeds
                except Exception as e:
                    raise ValueError(
                        f'{e} \ninput_embeds[selected].shape={input_embeds[img_bos + 1: img_eos].shape}, '
                        f'vit_embeds.shape={single_vit_embeds.shape}\n'
                        f'Please check whether shape of input_embeds[selected] matches the shape of vit_embeds.\n'
                        f'If not, please check whether self.img_context_token_id '
                        f'and the token-id of "<IMG_CONTEXT>" in processor are the same'
                    ) from e

        input_embeds = input_embeds.reshape(-1, embedding_size)
        return input_embeds

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
        # add dynamic
        self.max_seq_len_cached = max(self.max_position_embeddings, max_seq_len)
        # warm_up 阶段会传入max_seq_len=max_input_length，导致 max_seq_len_cached 开始就达到最大
        if self.rope_scaling is None:
            # RotaryEmbedding
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                             self.device,
                                                             self.max_position_embeddings)
        elif (self.scaling_type == "dynamic") and (self.max_seq_len_cached > self.max_position_embeddings):
            # DynamicNTKScalingRotaryEmbedding
            if self.max_position_embeddings == 0:
                raise ZeroDivisionError("Max position embeddings is zero")
            if self.dim == 2:
                raise ZeroDivisionError(
                    "When calculating RoPE base the divisor in the formula for power positions will be zero")
            base = self.base * (
                    np.divide(self.rope_scaling_factor * self.max_seq_len_cached, self.max_position_embeddings)
                    - (self.rope_scaling_factor - 1)
            ) ** (np.divide(self.dim, self.dim - 2))
            self.rotary_embedding = self.rotary_embedding.static(dim=self.head_size, base=base,
                                                                 device=self.rotary_embedding_device,
                                                                 scaling_factor=self.scaling_factor,
                                                                 ).to(self.device)
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                             self.device,
                                                             self.max_seq_len_cached)
        else:  # LinearScalingRotaryEmbedding
            # 如果 max_input_length > max_position_embeddings, 需要重置 base 和 rotary_embedding.inv_freq
            self.rotary_embedding = self.rotary_embedding.static(dim=self.head_size, base=self.base,
                                                                 device=self.rotary_embedding_device,
                                                                 scaling_factor=self.scaling_factor,
                                                                 ).to(self.device)
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                             self.device,
                                                             self.max_position_embeddings)

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        if is_prefill:
            if self.skip_word_embedding:
                if len(input_ids.shape) < 2:
                    input_ids = self.language_model.model.embed_tokens(input_ids)

            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
                                                          kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                          kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                               dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            input_tokens = self.placeholder if self.skip_word_embedding else input_ids
            input_embeddings = input_ids if self.skip_word_embedding else self.placeholder

            if self.dtype == torch.bfloat16:
                input_atten_mask = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            else:
                input_atten_mask = atten_mask
        else:
            input_tokens = input_ids
            input_embeddings = self.placeholder
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            if self.dtype == torch.bfloat16:
                input_atten_mask = torch.zeros(input_lengths.size(0),
                                               self.num_attention_heads,
                                               1, input_lengths.max(),
                                               dtype=self.dtype,
                                               device=self.device)
            else:
                input_atten_mask = self.attn_mask_fake

        self.acl_operation_inputs = [
            input_tokens,
            input_embeddings,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            input_atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32),
            lm_head_indices if is_prefill else self.lm_head_indices_fake,
        ]

        for ind, item in enumerate(self.acl_operation_inputs):
            logger.debug(f"{ind} {item.device=}")
        return self.acl_operation_inputs, self.acl_param

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
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        if is_prefill and input_ids.dim() == 1:
            input_ids = self.prepare_prefill_token_service(input_ids)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)

        return logits
