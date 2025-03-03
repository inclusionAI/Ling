# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import logging
import torch
import torch.nn as nn
from minigpt4.models.eva_vit import create_eva_vit_g
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.models.base_model import BaseModel


class MiniGPT4ImageEmbedding(BaseModel):
    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            freeze_vit=True,
    ):
        super().__init__()

        # Load VIT model
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, freeze_vit
        )

    @classmethod
    def init_vision_encoder(
            cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, freeze
    ):
        logging.info('Loading VIT')

        if not model_name == "eva_clip_g":
            logging.error('vit model must be eva_clip_g for current version of MiniGPT-4')

        visual_encoder = create_eva_vit_g(img_size, drop_path_rate, use_grad_checkpoint, 'fp32')

        ln_vision = LayerNorm(visual_encoder.num_features)
        if freeze:
            for _, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for _, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info("freeze vision encoder")

        logging.info('Loading VIT Done')
        return visual_encoder, ln_vision

    @classmethod
    def init_q_former(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("${model_path}/weights_image/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        q_former = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        q_former.cls = None
        q_former.bert.embeddings.word_embeddings = None
        q_former.bert.embeddings.position_embeddings = None
        for layer in q_former.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for _, param in q_former.named_parameters():
                param.requires_grad = False
            q_former = q_former.eval()
            q_former.train = disabled_train
            query_tokens.requires_grad = False

        return q_former, query_tokens

    def forward(self, image):
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)

        return image_embeds


def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

