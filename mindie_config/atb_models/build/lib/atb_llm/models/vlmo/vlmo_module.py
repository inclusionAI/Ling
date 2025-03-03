# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
import os
import logging as logger
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import vlmo.modules.multiway_transformer
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vlmo.modules import heads, objectives, vlmo_utils
from scipy import interpolate
from timm.models import create_model
from atb_speed.common.cpu_binding import CPUBinder
from vlmo.modules.vlmo_file_check import file_check, ErrorCode


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


def get_rank_and_world_size():
    try:
        rank = int(os.getenv("RANK", "0"))
    except ValueError:
        rank = 0
    try:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    return rank, world_size


RANK = 0
WORLD_SIZE = 2
IS_ND = False
TRANSFORMER_CONST = "transformer."
LOST_NAMES_CONST = "loss_names"
VLMO_FLASHATTENTIONMODEL_CONST = "vlmo_FlashAttentionModel"
TEXT_ENBEDDING_POSITION_EMBEDDING_WEIGHT = "text_embeddings.position_embeddings.weight"
TEXT_EMBEDDINGS_POSITION_IDS = "text_embeddings.position_ids"
TRANSFORMER_POS_ENBED = "transformer.pos_embed"
RELATIVE_POSITION_BIAS_TABLE = "relative_position_bias_table"
GAMMA_1 = "gamma_1"
GAMMA_2 = "gamma_2"
NORM1_WEIGHT = "norm1.weight"
NORM1_BIAS = "norm1.bias"
ATTN_QKV_BIAS = "attn.qkv_bias"
ATTN_Q_BIAS = "attn.q_bias"
ATTN_V_BIAS = "attn.v_bias"
ATTN_QKV_WEIGHT = "attn.qkv.weight"
ATTN_PROG_WEIGHT = "attn.proj.weight"
ATTN_PROJ_BIAS = "attn.proj.bias"
TOKENOFFSET = "tokenOffset"
SEQLEN_CONST = "seqLen"
IMAGE_CONST = "image"


def load_acl_transformer():
    """
    加载acl transformers
    :return:
    """
    acl_transformer_home_path = os.getenv("ATB_SPEED_HOME_PATH", "")
    if not acl_transformer_home_path or not os.path.exists(acl_transformer_home_path):
        logger.error("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh",
                    extra={'error_code': ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise RuntimeError("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(acl_transformer_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)


load_acl_transformer()


def round_up(x, align):
    if align == 0:
        return -1
    return (x + align - 1) // align * align


class KVAttentionManager:
    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        max_sequence_length,
        batch_size,
        seq_length,
        num_head
        
    ):
        self.nz_dim = 16
        self.is_full = True
        self.seq_length = seq_length  # 当前输入的最大长度，注意区分max_sequence_length
        self.batch_size = batch_size
        self.num_layers = num_hidden_layers
        self.num_head = num_head // WORLD_SIZE
        self.hidden_size = hidden_size // WORLD_SIZE
        self.max_sequence_length = max_sequence_length
        self.attention_mask_max_full = torch.full(
            (batch_size, self.max_sequence_length, self.max_sequence_length),
            torch.finfo(torch.half).min,
            dtype=torch.half,
        ).npu()
        self.seq_len_list_full = None
        self.seq_len_tensor_full = None
        self.token_offset_tensor = None
        self.seq_len_tensor_inc = None
        self.seq_len_list_inc = None
        self.attention_mask_max_inc = torch.full(
            (batch_size, self.max_sequence_length, self.max_sequence_length),
            0,
            dtype=torch.half,
        ).npu()
        if not IS_ND:
            self.k_cache_input = (
                torch.zeros(
                    self.num_layers,
                    batch_size,
                    self.hidden_size // self.nz_dim,
                    round_up(self.max_sequence_length, 16),
                    self.nz_dim,
                    device="npu", dtype=torch.half
                )
            )
            self.v_cache_input = self.k_cache_input.clone()
            self.k_cache_input = torch_npu.npu_format_cast(self.k_cache_input, 29)
            self.v_cache_input = torch_npu.npu_format_cast(self.v_cache_input, 29)
        else :            
            self.k_cache_input = (
                torch.zeros(
                    self.num_layers,
                    batch_size,
                    self.max_sequence_length,
                    self.hidden_size,
                    device="cpu",
                )
                .npu()
                .half()
                .contiguous()
            )
            self.v_cache_input = self.k_cache_input.clone()
        self.token_offset = 1

    @property
    def seq_len_tensor(self):
        if self.is_full:
            return self.seq_len_tensor_full
        return self.seq_len_tensor_inc

    @property
    def seq_len_list(self):
        if self.is_full:
            return self.seq_len_list_full
        return self.seq_len_list_inc

    @property
    def token_offset_list(self):
        return [self.token_offset] * self.batch_size

    def init_seq_len_and_token_offset(self, seq_len):
        self.token_offset = seq_len
        self.seq_len_list_full = [self.token_offset] * self.batch_size
        self.seq_len_tensor_full = torch.full(
            (self.batch_size,), self.token_offset, dtype=torch.int32
        ).npu()
        self.seq_len_list_inc = [1] * self.batch_size
        self.seq_len_tensor_inc = torch.full(
            (self.batch_size,), 1, dtype=torch.int32
        ).npu()

        self.token_offset_tensor = torch.full(
            (self.batch_size,), self.token_offset, dtype=torch.int32
        ).npu()

    def get_len_tensor(self, length):
        """
        :param length:
        :return:
        """
        return torch.full((self.batch_size,), length, dtype=torch.int32).npu()


def convert_to_textpt_ckpt(state_dict, module):
    new_state_dict = {}

    # Merge relative_position_bias_table from all layer into one tensor,
    # so we can use one op for gather the relative position bias for speed up
    relative_position_bias_tables = {}

    for key in state_dict:
        value = state_dict[key]

        if RELATIVE_POSITION_BIAS_TABLE in key:
            # transformer.blocks.0.attn.relative_position_bias_table
            layer_idx = int(key.split(".attn.")[0].split(".")[-1])
            relative_position_bias_tables[layer_idx] = value
            continue

        if "mlp" in key:
            key_imag = TRANSFORMER_CONST + key.replace("mlp", "mlp_imag")
            new_state_dict[key_imag] = value
        elif "norm2" in key:
            key_imag = TRANSFORMER_CONST + key.replace("norm2", "norm2_imag")
            new_state_dict[key_imag] = value
        else:
            new_key = TRANSFORMER_CONST + key
            new_state_dict[new_key] = value

    if len(relative_position_bias_tables) > 0:
        tensor_list = []
        for layer_idx in sorted(relative_position_bias_tables.keys()):
            tensor_list.append(relative_position_bias_tables[layer_idx])
        relative_position_bias_table = torch.cat(tensor_list, dim=1)

        num_distence, _ = relative_position_bias_table.shape
        all_relative_position_bias_table = (
            module.relative_position_bias_table.data.clone()
        )
        all_relative_position_bias_table[:num_distence, :] = (
            relative_position_bias_table
        )

        new_state_dict[RELATIVE_POSITION_BIAS_TABLE] = (
            all_relative_position_bias_table
        )

    return new_state_dict


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        logger.info("reshape position embedding from %d to %d", orig_size**2, new_size**2)

        return new_pos_embed
    else:
        return pos_embed_checkpoint


def convert_deepspeed_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("module."):
            new_key = key[len("module.") :]
            value = state_dict[key]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def bind_cpu(self):
    """
    绑核
    :return:
    """
    cpu_binder = CPUBinder()
    cpu_binder.bind_cpus(self.device_id_list, self.rank, 1.0)
    logger.info("Bind cpu successfully!")


class VLMo(nn.Module):
    def __init__(self, config):
        global IS_ND
        IS_ND = is_nd()
        global RANK
        global WORLD_SIZE
        RANK, WORLD_SIZE = get_rank_and_world_size()
        super().__init__()
        self.rank = RANK
        self.device_id_list = config['device']
        bind_cpu(self)


        # backbone & patch projection
        self.img_size = config["image_size"]
        self.transformer = create_model(
            config["model_arch"],
            world_size=WORLD_SIZE,
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            config=config,
        )
        self.patch_size = self.transformer.patch_size
        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)
        self.kv_attention_manager_vl = None
        self.kv_attention_manager_text = None
        self.kv_attention_manager_image = None
        self.batch_size_vl = 0
        self.batch_size_text = 0
        self.batch_size_image = 0
        self.image_masks = None

        # language embedding
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.num_features,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"],
            position_embedding_type=(
                "rel_pos"
                if self.transformer.need_relative_position_embed
                else "absolute"
            ),
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(objectives.init_weights)

        # task layers
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)

        ## language modeling
        if config[LOST_NAMES_CONST]["mlm"] > 0 or config[LOST_NAMES_CONST]["textmlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        ## image-text matching (global hard negative)
        if config[LOST_NAMES_CONST]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.num_features)
            self.itm_score.apply(objectives.init_weights)

        ## contrastive loss (or sampling for global hard negative)
        if config[LOST_NAMES_CONST]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.num_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ## retrieval task ft
        if config[LOST_NAMES_CONST]["irtr"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.load_pretrained_weight(config)

        # ===================== Downstream ===================== #
        ## VQAv2
        if config[LOST_NAMES_CONST]["vqa"] > 0:
            vs = config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        ## NLVR2 (Visual reasoning)
        if config[LOST_NAMES_CONST]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.num_features * 2, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, self.num_features)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            config_load_path = config["load_path"]
            file_check(config_load_path)
            ckpt = torch.load(config_load_path, map_location="cpu", weights_only=True)

            state_dict = None
            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                state_dict = ckpt
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
        # 把list移到初始化

        self.relative_position_bias_list_vl = self.get_rel_pos_bias(
            self.text_imag_relative_position_index
        )
        self.relative_position_bias_list_text = self.get_rel_pos_bias(
            self.text_relative_position_index
        )
        self.relative_position_bias_list_image = self.get_rel_pos_bias(
            self.relative_position_index
        )

        self.world_size = WORLD_SIZE
        self.init_ascend_operations(config)
        self.layer_id_list = [
            torch.tensor([i], dtype=torch.int32).npu()
            for i in range(self.transformer.depth)
        ]
        self.place_holder = torch.ones(1).npu()
    
    def trans_data(self, tensor):
        padding = (
            0, (round_up(tensor.size(3), 16) - tensor.size(3)), 
            0, (round_up(tensor.size(2), 16) - tensor.size(2)),
            0, 0,
            0, 0,
        )
        tensor = torch.nn.functional.pad(tensor, padding)
        return torch_npu.npu_format_cast(tensor.view(
            tensor.size(1), tensor.size(2),
            tensor.size(3) // 16, 16).transpose(1, 2).contiguous(), 29)

    def init_ascend_operations(self, config):
        self.acl_vl_param = json.dumps(
            {
                "rmsNormEps": 1e-5,
                "headNum": self.transformer.num_heads // WORLD_SIZE,
                "dk": self.transformer.embed_dim // self.transformer.num_heads,
                "layerNum": self.transformer.depth,
                "rank": self.rank,
                "rankSize": self.world_size,
                "backend": "lccl" if IS_ND else "hccl",
                "maxTextLen": config["max_text_len"],
                "vlLayerIndex": self.transformer.vlffn_start_layer_index,
            }
        )
        self.acl_others_param = json.dumps(
            {
                "rmsNormEps": 1e-5,
                "headNum": self.transformer.num_heads // WORLD_SIZE,
                "dk": self.transformer.embed_dim // self.transformer.num_heads,
                "layerNum": self.transformer.depth,
                "rank": self.rank,
                "rankSize": self.world_size,
                "backend": "lccl" if IS_ND else "hccl",
                "maxTextLen": config["max_text_len"],
                "vlLayerIndex": 0,
            }
        )
        self.acl_fa_vl_operation = torch.classes.ModelTorch.ModelTorch(
            VLMO_FLASHATTENTIONMODEL_CONST
        )
        self.acl_fa_vl_operation.set_param(self.acl_vl_param)

        self.acl_fa_text_operation = torch.classes.ModelTorch.ModelTorch(
            VLMO_FLASHATTENTIONMODEL_CONST
        )
        self.acl_fa_text_operation.set_param(self.acl_others_param)

        self.acl_fa_image_operation = torch.classes.ModelTorch.ModelTorch(
            VLMO_FLASHATTENTIONMODEL_CONST
        )
        self.acl_fa_image_operation.set_param(self.acl_others_param)

        self.num_layers = self.transformer.depth
        self.hidden_size = self.transformer.embed_dim
        self.ascend_weight_vl = []
        self.ascend_weight_text = []
        self.ascend_weight_image = []
        self.lm_head_weight = None
        self.batch_size = 0
        self.kv_attention_manager_vl = None
        self.kv_attention_manager_text = None
        self.kv_attention_manager_image = None


    def load_pretrained_weight(self, config):
        if (
            config["load_path"] != ""
            and not config["test_only"]
        ):
            config = config
            tmp_config_load_path = file_check(config["load_path"])
            ckpt = torch.load(tmp_config_load_path, map_location="npu", weights_only=True)


            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                state_dict = ckpt

            if config[LOST_NAMES_CONST]["textmlm"] > 0:
                state_dict = convert_to_textpt_ckpt(state_dict, self)

            max_text_len = config["max_text_len"]
            if (
                TEXT_ENBEDDING_POSITION_EMBEDDING_WEIGHT in state_dict
                and TEXT_EMBEDDINGS_POSITION_IDS in state_dict
                and state_dict[TEXT_ENBEDDING_POSITION_EMBEDDING_WEIGHT].size(0)
                != max_text_len
            ):
                state_dict[TEXT_ENBEDDING_POSITION_EMBEDDING_WEIGHT].data = (
                    state_dict[TEXT_ENBEDDING_POSITION_EMBEDDING_WEIGHT].data[
                        :max_text_len, :
                    ]
                )
                state_dict[TEXT_EMBEDDINGS_POSITION_IDS].data = state_dict[
                    TEXT_EMBEDDINGS_POSITION_IDS
                ].data[:, :max_text_len]
                for check_key in (
                    "relative_position_index",
                    "text_relative_position_index",
                    "text_imag_relative_position_index",
                ):
                    if check_key in state_dict:
                        state_dict.pop(check_key)

            if TRANSFORMER_POS_ENBED in state_dict:
                pos_embed_reshaped = interpolate_pos_embed(
                    state_dict[TRANSFORMER_POS_ENBED], self.transformer
                )
                state_dict[TRANSFORMER_POS_ENBED] = pos_embed_reshaped

            if RELATIVE_POSITION_BIAS_TABLE in state_dict:
                rel_pos_bias = state_dict[RELATIVE_POSITION_BIAS_TABLE]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.relative_position_bias_table.size()
                dst_patch_shape = self.transformer.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    logger.error("dst_patch_shape[0] is not equal dst_patch_shape[1]",
                                extra={'error_code': ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                    dst_patch_shape[1] * 2 - 1
                )
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    q = None
                    state_dict.pop("relative_position_index")
                    state_dict.pop("text_relative_position_index")
                    state_dict.pop("text_imag_relative_position_index")

                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        try:
                            return a * (1.0 - r**n) / (1.0 - r)
                        except ZeroDivisionError:
                            logger.info("The divisor cannot be 0.")
                            return 1.0

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q
                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)



                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind="cubic")
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy))
                            .contiguous()
                            .view(-1, 1)
                            .to(rel_pos_bias.device)
                        )

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    state_dict[RELATIVE_POSITION_BIAS_TABLE] = new_rel_pos_bias

            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )


    def get_rel_pos_bias(self, relative_position_index):
        if self.relative_position_embed:
            relative_position_bias = F.embedding(
                relative_position_index.long().to(
                    self.relative_position_bias_table.device
                ),
                self.relative_position_bias_table,
            )
            all_relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, x, y
            relative_position_bias_list = torch.chunk(
                all_relative_position_bias, self.num_layers, dim=0
            )
            res = []
            for i in range(self.num_layers):
                res.append(relative_position_bias_list[i].unsqueeze(0).half().npu().chunk(WORLD_SIZE, dim=1))
                
            return res
        else:
            return [None] * self.num_layers


    def build_relative_position_embed(self, config):
        if not self.transformer.need_relative_position_embed:
            self.relative_position_embed = False
            self.text_imag_relative_position_index = None
            self.text_relative_position_index = None
            self.relative_position_index = None
            return
        self.relative_position_embed = True
        window_size = (
            int(self.img_size / self.patch_size),
            int(self.img_size / self.patch_size),
        )  # (14, 14)
        num_heads = self.transformer.num_heads
        max_text_len_of_initckpt = config["max_text_len_of_initckpt"]  # 196
        max_text_len = config["max_text_len"]  # 40
        max_imag_len = window_size[0] * window_size[1] + 1  # 197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = (
            self.num_relative_distance + self.text_num_relative_distance + 2
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index

        text_position_ids = torch.arange(max_text_len - 1)
        text_rel_pos_mat = text_position_ids.unsqueeze(
            -2
        ) - text_position_ids.unsqueeze(-1)
        min_distance = int(2 - max_text_len_of_initckpt)  # -194
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += self.num_relative_distance + 2
        text_relative_position_index = torch.zeros(
            size=(max_text_len,) * 2, dtype=relative_coords.dtype
        )
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index

        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (
            self.num_relative_distance
        )
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (
            self.num_relative_distance + 1
        )

        text_row_relative_position_index = torch.cat(
            (text_relative_position_index, text2imag_relative_position_index), 1
        )
        imag_row_relative_position_index = torch.cat(
            (imag2text_relative_position_index, relative_position_index), 1
        )
        text_imag_relative_position_index = torch.cat(
            (text_row_relative_position_index, imag_row_relative_position_index), 0
        )
        self.text_imag_relative_position_index = text_imag_relative_position_index
    
    def add_specified_string(self, weights_t, str_keys, add_string_list, weights_layer):
        for item in add_string_list:
            weights_t.append(weights_layer[str_keys + item])

    def init_acl_weight(self, modality_type="vl"):
        weights = []

        empty_k_bias = torch.zeros(
            self.transformer.embed_dim // WORLD_SIZE, device="npu", dtype=torch.float16
        )
        weights_layer = self.state_dict()
 
        if modality_type == "vl":
            for i in range(self.transformer.depth):
                str_keys = f"transformer.blocks.{i}."
                weights_t = []
                add_string_list_1 = [GAMMA_1, GAMMA_2, NORM1_WEIGHT, NORM1_BIAS]
                add_string_list_2 = [ATTN_QKV_BIAS, ATTN_QKV_WEIGHT, ATTN_PROG_WEIGHT, ATTN_PROJ_BIAS]
                add_string_list = add_string_list_1
                self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)
                weights_layer[str_keys + ATTN_QKV_BIAS] = torch.cat(
                    [torch.cat([weights_layer[str_keys + ATTN_Q_BIAS],
                    empty_k_bias], dim=0), weights_layer[str_keys + ATTN_V_BIAS]], dim=0)
                add_string_list = add_string_list_2 
                self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)
                if i < self.vlffn_start_layer_index:
                    add_string_list_3 = [
                        "norm2_text.weight", "norm2_text.bias", "norm2_imag.weight", "norm2_imag.bias", 
                        "mlp_text.fc1.weight", "mlp_text.fc2.weight", "mlp_text.fc1.bias", "mlp_text.fc2.bias",
                        "mlp_imag.fc1.weight", "mlp_imag.fc2.weight", "mlp_imag.fc1.bias", "mlp_imag.fc2.bias"
                    ]
                    add_string_list = add_string_list_3
                    self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)
                else:
                    weights_t.append(weights_layer[str_keys + "norm2_vl.weight"])
                    weights_t.append(weights_layer[str_keys + "norm2_vl.bias"])
                    weights_t.append(weights_layer[str_keys + "mlp_vl.fc1.weight"])
                    weights_t.append(weights_layer[str_keys + "mlp_vl.fc2.weight"])
                    weights_t.append(weights_layer[str_keys + "mlp_vl.fc1.bias"])
                    weights_t.append(weights_layer[str_keys + "mlp_vl.fc2.bias"])

                weights.extend(weights_t)
            self.ascend_weight_vl = weights
            self.acl_fa_vl_operation.set_weight(weights)
        elif modality_type == IMAGE_CONST:
            for i in range(self.transformer.depth):
                str_keys = f"transformer.blocks.{i}."
                weights_t = []
                add_string_list_4 = [GAMMA_1, GAMMA_2, NORM1_WEIGHT, NORM1_BIAS]
                add_string_list_5 = [
                    ATTN_QKV_BIAS, ATTN_QKV_WEIGHT, ATTN_PROG_WEIGHT, ATTN_PROJ_BIAS,
                    "norm2_imag.weight", "norm2_imag.bias", "mlp_imag.fc1.weight", "mlp_imag.fc2.weight",
                    "mlp_imag.fc1.bias", "mlp_imag.fc2.bias"
                ]
                add_string_list = add_string_list_4
                self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)
                weights_layer[str_keys + ATTN_QKV_BIAS] = torch.cat(
                    [torch.cat([weights_layer[str_keys + ATTN_Q_BIAS],
                    empty_k_bias], dim=0), weights_layer[str_keys + ATTN_V_BIAS]], dim=0) 
                add_string_list = add_string_list_5
                self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)                
                weights.extend(weights_t)

            self.ascend_weight_image = weights
            self.acl_fa_image_operation.set_weight(weights)
        elif modality_type == "text":

            for i in range(self.transformer.depth):
                str_keys = f"transformer.blocks.{i}."
                weights_t = []
                add_string_list_6 = [GAMMA_1, GAMMA_2, NORM1_WEIGHT, NORM1_BIAS]
                add_string_list_7 = [
                    ATTN_QKV_BIAS, ATTN_QKV_WEIGHT, ATTN_PROG_WEIGHT, ATTN_PROJ_BIAS,
                    "norm2_text.weight", "norm2_text.bias", "mlp_text.fc1.weight", "mlp_text.fc2.weight",
                    "mlp_text.fc1.bias", "mlp_text.fc2.bias"
                ]
                add_string_list = add_string_list_6
                self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)
                weights_layer[str_keys + ATTN_QKV_BIAS] = torch.cat(
                    [torch.cat([weights_layer[str_keys + ATTN_Q_BIAS],
                    empty_k_bias], dim=0), weights_layer[str_keys + ATTN_V_BIAS]], dim=0) 
                add_string_list = add_string_list_7
                self.add_specified_string(weights_t, str_keys, add_string_list, weights_layer)   
                weights.extend(weights_t)
            self.ascend_weight_text = weights
            self.acl_fa_text_operation.set_weight(weights)


    def init_acl_encoder_param(self, x, mask=None, modality_type="vl"):
        x = x.half().npu()
        mask_bool = mask.to(dtype=bool, device=x.device)
        mask_list = []
        if modality_type == "vl":
            for i in range(self.transformer.depth): 
                mask_bias = self.relative_position_bias_list_vl[i][RANK].masked_fill(~mask_bool[:, None, None, :],
                            -10000)
                if not IS_ND:
                    mask_bias = self.trans_data(mask_bias)
                else:
                    mask_bias = mask_bias
                mask_list.append(mask_bias)
            inputs = [
                x,
                self.kv_attention_manager_vl.k_cache_input,
                self.kv_attention_manager_vl.v_cache_input,
                self.kv_attention_manager_vl.token_offset_tensor,
                self.kv_attention_manager_vl.seq_len_tensor,
                self.place_holder,
            ] + self.layer_id_list + mask_list
           
            return inputs
        elif modality_type == IMAGE_CONST:
            for i in range(self.transformer.depth):
                mask_bias = (self.relative_position_bias_list_image[i][RANK].
                             masked_fill(~mask_bool[:, None, None, :], -10000))
                if not IS_ND:
                    mask_bias = self.trans_data(mask_bias)
                mask_list.append(mask_bias)
            inputs = [
                x,
                self.kv_attention_manager_image.k_cache_input,
                self.kv_attention_manager_image.v_cache_input,
                self.kv_attention_manager_image.token_offset_tensor,
                self.kv_attention_manager_image.seq_len_tensor,
                self.place_holder,
            ] + self.layer_id_list + mask_list
            return inputs
        else:
            for i in range(self.transformer.depth):
                mask_bias = self.relative_position_bias_list_text[i][RANK].masked_fill(~mask_bool[:, None, None, :],
                            -10000)
                if not IS_ND:
                    mask_bias = self.trans_data(mask_bias)
                mask_list.append(mask_bias)
            inputs = [
                x,
                self.kv_attention_manager_text.k_cache_input,
                self.kv_attention_manager_text.v_cache_input,
                self.kv_attention_manager_text.token_offset_tensor,
                self.kv_attention_manager_text.seq_len_tensor,
                self.place_holder,
            ] + self.layer_id_list + mask_list
            return inputs


    def execute_acl_encoder(self, x, mask, modality_type):
        acl_input = self.init_acl_encoder_param(x, mask, modality_type)
        acl_model_out = None
        if modality_type == "vl":
            tmp_param = json.dumps(
                {
                    TOKENOFFSET: self.kv_attention_manager_vl.token_offset_list,
                    SEQLEN_CONST: self.kv_attention_manager_vl.seq_len_list,
                }
            )
            acl_model_out = self.acl_fa_vl_operation.execute(acl_input, tmp_param)
            acl_model_out = acl_model_out[0]
        elif modality_type == IMAGE_CONST:
            tmp_param = json.dumps(
                {
                    TOKENOFFSET: self.kv_attention_manager_image.token_offset_list,
                    SEQLEN_CONST: self.kv_attention_manager_image.seq_len_list,
                }
            )
            acl_model_out = self.acl_fa_image_operation.execute(acl_input, tmp_param)
            acl_model_out = acl_model_out[0]
        elif modality_type == "text":
            tmp_param = json.dumps(
                {
                    TOKENOFFSET: self.kv_attention_manager_text.token_offset_list,
                    SEQLEN_CONST: self.kv_attention_manager_text.seq_len_list,
                }
            )
            acl_model_out = self.acl_fa_text_operation.execute(acl_input, tmp_param)
            acl_model_out = acl_model_out[0]
        return acl_model_out


    def infer_ascend(
        self,
        batch,
        mask_text=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = IMAGE_CONST

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch["text_masks"]

        text_embeds = self.text_embeddings(text_ids)

        img = batch[imgkey][0].half()

        image_embeds, image_masks = self.transformer.visual_embed(img)
        if self.image_masks is None:
            self.image_masks = image_masks.long().to(device=img.device)
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(self.image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, self.image_masks], dim=1)

        x = co_embeds
        # F.embedding

        batch_size = x.shape[0]
        input_shape = x.shape[1]
        if batch_size != self.batch_size_vl:
            self.batch_size_vl = batch_size
            self.kv_attention_manager_vl = KVAttentionManager(
                self.transformer.depth,
                self.transformer.embed_dim,
                input_shape,
                batch_size,
                input_shape,
                self.transformer.num_heads,
                
            )
        self.kv_attention_manager_vl.init_seq_len_and_token_offset(
            input_shape
        )  # for ascend

        if not self.ascend_weight_vl:
            self.init_acl_weight(modality_type="vl")
        acl_x = self.execute_acl_encoder(co_embeds, co_masks, modality_type="vl")
        x = acl_x

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        # nn.Tanh() nn.Linear()
        cls_feats = self.pooler(x)
        ################################################################################################
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            IMAGE_CONST: img,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_ft_ascend(
        self,
        batch,
        mask_text=False,
    ):
        logger.info("vlmo module infer text ft ascend")
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch["text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(text_masks)
        )

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds

        batch_size = x.shape[0]
        input_shape = x.shape[1]

        if batch_size != self.batch_size_text:
            self.batch_size_text = batch_size
            self.kv_attention_manager_text = KVAttentionManager(
                self.transformer.depth,
                self.transformer.embed_dim,
                input_shape,
                batch_size,
                input_shape,
                self.transformer.num_heads,
            )
        self.kv_attention_manager_text.init_seq_len_and_token_offset(
            input_shape
        )  # for ascend

        if not self.ascend_weight_text:
            self.init_acl_weight(modality_type="text")
        acl_x = self.execute_acl_encoder(co_embeds, co_masks, modality_type="text")

        lffn_hiddens = self.transformer.norm(acl_x)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret


    def infer_image(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        logger.info("vlmo module infer image")
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = IMAGE_CONST

        img_infer = batch[imgkey][0]
        image_embeds, image_masks = self.transformer.visual_embed(img_infer)

        image_masks = image_masks.long().to(device=img_infer.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(
            self.relative_position_index
        )

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(
                x,
                mask=co_masks,
                modality_type=IMAGE_CONST,
                relative_position_bias=relative_position_bias_list[i],
            )
            all_hidden_states.append(x)

        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index - 1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](
                vlffn_hiddens,
                mask=co_masks,
                modality_type="vl",
                relative_position_bias=relative_position_bias_list[vlffn_index],
            )

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret


    def infer_image_ft(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        logger.info("vlmo module infer_image_ft")
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = IMAGE_CONST

        img_ft = batch[imgkey][0]
        image_embeds, image_masks = self.transformer.visual_embed(img_ft)

        image_masks = image_masks.long().to(device=img_ft.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )

        co_embeds = image_embeds
        co_masks = image_masks

        x_ft = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(
            self.relative_position_index
        )

        for i, blk in enumerate(self.transformer.blocks):
            x_ft = blk(
                x_ft,
                mask=co_masks,
                modality_type=IMAGE_CONST,
                relative_position_bias=relative_position_bias_list[i],
            )
            all_hidden_states.append(x_ft)

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )
        logger.info("vlmo module infer image_ft")
        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x_ft[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret


    def infer_image_ft_ascend(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        logger.info("vlmo module infer_image_ft_ascend")
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = IMAGE_CONST

        img_ascend = batch[imgkey][0].reshape(1, 3, 384, 384)

        image_embeds, image_masks = self.transformer.visual_embed(img_ascend)
        image_masks = image_masks.long().to(device=img_ascend.get_device())

        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        batch_size = x.shape[0]
        input_shape = x.shape[1]

        if batch_size != self.batch_size_image:
            self.batch_size_image = batch_size
            self.kv_attention_manager_image = KVAttentionManager(
                self.transformer.depth,
                self.transformer.embed_dim,
                input_shape,
                batch_size,
                input_shape,
                self.transformer.num_heads,
            )
        self.kv_attention_manager_image.init_seq_len_and_token_offset(
            input_shape
        )  # for ascend

        if not self.ascend_weight_image:
            self.init_acl_weight(modality_type=IMAGE_CONST)
        acl_x = self.execute_acl_encoder(co_embeds, co_masks, modality_type=IMAGE_CONST)

        vffn_hiddens = self.transformer.norm(acl_x)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret


    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Textonly Masked Language Modeling
        if "textmlm" in self.current_tasks:
            ret.update(objectives.compute_textonly_mlm(self, batch))

        # Contrastive loss for pretraining
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # Contrastive loss for finetuning
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        # Image Text Matching with global hard negative, must use with itc
        if "itm" in self.current_tasks:
            itc_i2t_logits = None
            itc_t2i_logits = None
            try:
                itc_i2t_logits = ret["itc_i2t_logits"]
                itc_t2i_logits = ret["itc_t2i_logits"]
            except KeyError:
                logger.info("itc_i2t_logits or itc_t2i_logits not exist")
            ret.update(
                objectives.compute_itm_hardneg(
                    self, batch, itc_i2t_logits, itc_t2i_logits
                )
            )

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        if self.world_size >= 2:
            torch.distributed.all_reduce(ret, op=torch.distributed.ReduceOp.SUM)
        return ret


    def training_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss


    def training_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)


    def validation_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        _ = self(batch)


    def validation_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)


    def test_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.config[LOST_NAMES_CONST]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret





    def configure_optimizers(self):
        return vlmo_utils.set_schedule(self)
