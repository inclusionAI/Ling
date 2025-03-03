# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import math
import re
from functools import partial, reduce
from typing import Optional, List, Tuple, Dict
from PIL import Image

import torch
from torch import nn
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (convert_to_rgb, normalize, rescale, resize, to_channel_dimension_format, )
from transformers.image_utils import (ChannelDimension, PILImageResampling, to_numpy_array, )

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.log.logging import logger
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from .modeling_siglipvision import SigLipVisionModel
from .modeling_bunny import FlashBunnyModel, BunnyConfig
from .configuration_bunny import SigLipVisionConfig


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
CPP_QWEN_MODEL_CLASS_NAME = "qwen_QwenDecoderModel"


class SigLipImageProcessor:
    def __init__(self,
                 image_mean=(0.5, 0.5, 0.5),
                 image_std=(0.5, 0.5, 0.5),
                 size=(384, 384),
                 crop_size: Dict[str, int] = None,
                 resample=PILImageResampling.BICUBIC,
                 rescale_factor=1 / 255,
                 data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            pass

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class FlashBunnyForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.transformer = FlashBunnyModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True
        )

        self.config = config
        self.attn_mask_fake = self.attn_mask.get_attn_mask(
            1, dtype=self.dtype, device="npu"
        )
        self.place_holder = torch.tensor([1], dtype=self.dtype, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch(
            "TransdataOperation"
        )
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        # visual
        self.vision_config = SigLipVisionConfig()
        self.visual = SigLipVisionModel(self.vision_config).to(
            device=weights.device, dtype=weights.dtype
        )
        del self.visual.vision_model.encoder.layers[-1:]
        self.visual.vision_model.head = nn.Identity()
        self.visual.requires_grad_(False)
        self.visual.eval()

        # vison_projector
        self.mlp1 = self.build_vision_projector(self.config)
        self.mlp1 = self.mlp1.to(device=weights.device, dtype=weights.dtype)

        self.init_module_weight(self.visual, weights, prefix="model.vision_tower.vision_tower")
        self.init_module_weight(self.mlp1, weights, prefix="model.mm_projector")

        self.acl_operation_inputs = None
        self.sin_table = None
        self.cos_table = None
        self.ascend_weight = None

    def build_vision_projector(self, config):
        projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    def init_module_weight(self, module, weights, prefix="vision_model"):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"{prefix}.{model_weight}"), requires_grad=False
            )
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    def init_ascend_operations(self, config: BunnyConfig):
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("Bunny_qwen2 does not support GQA")

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_QWEN_MODEL_CLASS_NAME)
        logger.info(">>>> bunny_qwen2_DecoderModel is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=None,
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=None,
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(
            self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper
        )
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        acl_param_dict = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "skipWordEmbedding": False,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "normEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "normType": NormType.RMS_NORM,
            "isUnpadInputs": True,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers,
        }
        acl_param_encoder = json.dumps(
            {**acl_param_dict, "isPrefill": True, "enableLcoc": self.lcoc_enable,
             "skipWordEmbedding": True}
        )
        acl_param_decoder = json.dumps(
            {**acl_param_dict, "isPrefill": False, "enableLcoc": False}
        )

        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, model_cfg):
        image_processor = SigLipImageProcessor()
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = self.expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        else:
            return image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def encode_images(self, images):
        # visual
        image_forward_outs = self.visual(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
        # mm_projector
        image_features = self.mlp1(image_features)
        return image_features

    def prepare_prefill_token(self, text, image, video, processor, batch_size):
        text_prompt = f"A chat between a curious user and an artificial intelligence assistant. \
        The assistant gives helpful, detailed, and polite answers to the user's questions. \
        USER: <image>\n{text} ASSISTANT:"
        text_chunks = [processor(chunk).input_ids for chunk in text_prompt.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1],
                                 dtype=torch.long).unsqueeze(0).to(self.device)
        if image is None or input_ids.shape[1] == 1:
            return self.transformer.wte(input_ids)
        image = safe_open_image(Image, image)
        images = self.process_images([image], self.config).to(dtype=self.dtype, device=self.device)
        image.close()
        if isinstance(images, list) or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_attention = zip(input_ids, attention_mask)
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in input_attention]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.transformer.wte(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.transformer.wte(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_padded = []

        for cur_new_embed in new_input_embeds:
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        return new_input_embeds[0]

    def prepare_inputs_for_ascend(
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
        **kwargs):
        acl_param = json.dumps({"seqLen": input_lengths.tolist()})
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype, self.device, self.max_position_embeddings
        )

        self.cos_table = self.rotary_embedding.get_cos_cached_total()
        self.sin_table = self.rotary_embedding.get_sin_cached_total()
        if is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(
                    range(input_ids.shape[0]),
                    dtype=torch.int64,
                    device=input_ids.device
                )

        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                attention_mask = self.attn_mask.get_attn_mask(
                    pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            else:
                attention_mask = self.attn_mask.get_attn_mask(
                    self.max_base_len, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
        else:
            attention_mask = self.attn_mask_fake

        self.acl_operation_inputs = [
            input_ids,
            position_ids,
            self.cos_table,
            self.sin_table,
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.place_holder,
            self.place_holder,
            self.place_holder,
            input_lengths.to(torch.int32),
            (
                lm_head_indices if is_prefill else self.lm_head_indices_fake
            ),
        ]

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
            **kwargs) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)

        acl_inputs, acl_param = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            **kwargs
        )

        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits