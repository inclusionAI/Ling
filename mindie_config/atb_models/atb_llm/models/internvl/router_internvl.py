# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

from dataclasses import dataclass
import torch

from atb_llm.models.base.router import BaseRouter
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.internvl.config_internvl import InternvlConfig
from atb_llm.models.internvl.data_preprocess_internvl import process_image_input, process_video_input
from atb_llm.models.internvl.input_builder_internvl import InternvlInputBuilder, INTERNVL_SYSTEM_PROMPTS
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger


_IMAGE = "image"
_VIDEO = "video"
_TEXT = "text"

# 服务化支持torch分布式所需环境变量
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '5678')


@dataclass
class InternvlRouter(BaseRouter):
    @property
    def model_version(self):
        """
        次级模型： v1=Llama, v2=InternLM2......
                  v3=Qwen 待支持
        :return:
        """
        try:
            llm_model_architectures = self.config_dict['llm_config']['architectures'][0]
        except KeyError as e:
            logger.error("`llm_config.architectures` does not exist! Check `config.json`.",
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError("`llm_config.architectures` does not exist! Check `config.json`.") from e

        if llm_model_architectures == 'LlamaForCausalLM':
            model_ver = 'v1'
        elif llm_model_architectures == 'InternLM2ForCausalLM':
            model_ver = "v2"
        else:
            logger.error("`llm_config.architectures` must in ['InternLM2ForCausalLM', 'LlamaForCausalLM']",
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError("`llm_config.architectures` must in ['InternLM2ForCausalLM', 'LlamaForCausalLM']")
        return model_ver

    def get_config(self):
        config = InternvlConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        config.model_name_or_path = self.model_name_or_path
        super().check_config(config)
        return config

    def get_tokenizer(self):
        try:
            llm_model_architectures = self.config_dict['llm_config']['architectures'][0]
        except KeyError as e:
            logger.error("`llm_config.architectures` does not exist! Check `config.json`.",
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError("`llm_config.architectures` does not exist! Check `config.json`.") from e
        
        if llm_model_architectures == 'LlamaForCausalLM':
            tokenizer = safe_get_tokenizer_from_pretrained(
                self.model_name_or_path,
                revision=self.revision,
                padding_side="left",
                trust_remote_code=self.trust_remote_code,
                use_fast=False
            )
        elif llm_model_architectures == 'InternLM2ForCausalLM':
            tokenizer = safe_get_tokenizer_from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code
            )
        else:
            logger.error("`llm_config.architectures` must in ['InternLM2ForCausalLM', 'LlamaForCausalLM'], "
                         f"got {llm_model_architectures}",
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError("`llm_config.architectures` must in ['InternLM2ForCausalLM', 'LlamaForCausalLM'], "
                             f"got {llm_model_architectures}")
        return tokenizer

    def get_input_builder(self):
        return InternvlInputBuilder(self.tokenizer, self.config)

    def tokenize(self, inputs, **kwargs):
        img_begin_id = self.tokenizer.encode("<img>")[-1]
        img_end_id = self.tokenizer.encode("</img>")[-1]
        shm_name_save_path = kwargs.get("shm_name_save_path", None)

        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        if patch_size == 0:
            logger.error('The vision patch_size of config can not be 0.',
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError('The vision patch_size of config can not be 0.')
        num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))
        
        use_dynamic_prepro = False if self.config.ps_version == "v1" else True

        system_prompt = INTERNVL_SYSTEM_PROMPTS[self.config.ps_version][self.config.template]
        query = ('<|im_start|>system\n'
                f'{system_prompt}<|im_end|><|im_start|>user\n')

        text = ""
        image_index = 1
        shm_name_list = []
        shape_value_list = []
        image_num = sum(1 for d in inputs if _IMAGE in d)
        for single_input in inputs:
            if _TEXT in single_input:
                text += single_input.get(_TEXT)
                continue
            if _IMAGE in single_input:
                current_query, shm_name_value, shape_value = process_image_input(
                    single_input,
                    image_num,
                    image_index,
                    use_dynamic_prepro,
                    num_image_token,
                    shm_name_save_path
                )
                query += current_query
                image_index += 1
                shm_name_list.append(shm_name_value)
                shape_value_list.append(shape_value)
            elif _VIDEO in single_input:
                current_query, shm_name_value, shape_value = process_video_input(
                    single_input,
                    use_dynamic_prepro,
                    num_image_token,
                    shm_name_save_path
                )
                query += current_query
                shm_name_list += shm_name_value
                shape_value_list += shape_value
            else:
                logger.error("Unsupport element, it should be a video or image, please check your input",
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise KeyError("Unsupport element, it should be a video or image, please check your input")
        query += f'{text}<|im_end|><|im_start|>assistant\n'
        query_ids = torch.tensor(self.tokenizer.encode(query))
        bos_pos_set = torch.nonzero(query_ids == img_begin_id).view(-1)
        eos_pos_set = torch.nonzero(query_ids == img_end_id).view(-1)
        for i, (bos_pos, eos_pos) in enumerate(zip(bos_pos_set, eos_pos_set)):
            if eos_pos - bos_pos < 3:
                logger.error("tokenize input error.",
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError("tokenize input error.")
            query_ids[bos_pos + 1] = shm_name_list[i]
            query_ids[bos_pos + 2] = shape_value_list[i]
            
        return query_ids