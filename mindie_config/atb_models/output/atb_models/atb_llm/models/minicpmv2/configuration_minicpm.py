# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MiniCPM model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode

logger = logging.get_logger(__name__)

MINICPM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class MiniCPMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniCPMModel`].
    ```"""

    model_type = "minicpm"
    keys_to_ignore_at_inference = ["past_key_values"]

    EOS_TOKEN_ID = "eos_token_id"
    TIE_WORD_EMBEDDINGs = "tie_word_embeddings"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        scale_emb=1,
        dim_model_base=1,
        scale_depth=1,
        pe_type='ROPE',
        rope_keep_local_base_windows=None,
        rope_vanilla_theta=None,
        rope_mscale=None,
        rope_given_inv_feq_str=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.pe_type = pe_type
        self.rope_keep_local_base_windows = rope_keep_local_base_windows
        self.rope_vanilla_theta = rope_vanilla_theta
        self.rope_mscale = rope_mscale
        self.rope_given_inv_feq_str = rope_given_inv_feq_str
        
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, 
            tie_word_embeddings=tie_word_embeddings, **kwargs,)

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            logger.error("error: rope_scaling in config is incorrect, please check config file", 
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            logger.error("error: rope_scaling type incorrect, check config", 
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )


class MiniCPMVConfig(MiniCPMConfig):
    model_type = "minicpmv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_encoder="vit_so400m_patch14_siglip_384.webli",
        query_num=64,
        image_size=448,
        drop_vision_last_layer=True,
        slice_mode=True,
        patch_size=14,
        max_slice_nums=9,
        scale_resolution=448,
        im_start_token_id=101,
        im_end_token_id=102,
        slice_start_token_id=111,
        slice_end_token_id=112,
        unk_token_id=0,
        **kwargs,
    ):
        self.vision_encoder = vision_encoder
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.slice_mode = slice_mode
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.slice_start_token_id = slice_start_token_id
        self.slice_end_token_id = slice_end_token_id
        self.unk_token_id = unk_token_id
        super().__init__(**kwargs)