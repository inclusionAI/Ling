from typing import Optional
from enum import Enum
import importlib
import os

from atb_llm.models.base.model_utils import safe_get_config_dict
from atb_llm.utils import file_utils


class InferenceMode(int, Enum):
    REGRESSION = 0
    SPECULATE = 1
    SPLITFUSE = 2
    PREFIXCACHE = 3


def get_model(model_name_or_path: str,
              is_flash_causal_lm: bool = True,
              load_tokenizer: bool = True,
              max_position_embeddings: Optional[int] = None,
              revision: Optional[str] = None,
              tokenizer_path: Optional[str] = None,
              trust_remote_code: bool = False,
              enable_atb_torch: bool = False):
    model_name_or_path = file_utils.standardize_path(model_name_or_path, check_link=False)
    file_utils.check_path_permission(model_name_or_path)
    model_type_key = 'model_type'
    config_dict = safe_get_config_dict(model_name_or_path)
    config_dict[model_type_key] = config_dict[model_type_key].lower()
    model_type = config_dict[model_type_key]
    if model_type == "kclgpt":
        model_type = "codeshell"
    elif model_type == "internvl_chat":
        model_type = "internvl"
    elif model_type == "llava_next_video":
        model_type = "llava_next"
    elif model_type == "llava" and "_name_or_path" in config_dict.keys():
        if "yi-vl" in config_dict["_name_or_path"].lower():
            model_type = config_dict[model_type_key] = "yivl"
    elif model_type == "minicpmv" and "MiniCPM-Llama3-V-2_5" in model_name_or_path:
        model_type = "minicpm_llama3_v2"
    elif "clip" in model_type:
        model_type = "clip"
    elif model_type == "bunny-qwen2" or model_type == "bunny-minicpm":
        model_type = "bunny"
    elif model_type == "chatglm" and "vision_config" in config_dict:
        model_type = "glm4v"
    elif model_type == "bailing_moe":
        model_type = "deepseek"


    # 安全校验
    current_path = os.path.dirname(os.path.abspath(__file__))
    supported_models = []
    for foldername in file_utils.safe_listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)

    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type}；"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )

    router_path = f"atb_llm.models.{model_type}.router_{model_type}"
    if model_type == "qwen2_moe" or model_type == "minicpm_llama3_v2":
        model_type = model_type.replace('_', '')
    if model_type == "qwen2_audio":
        model_type = model_type.replace('_', '')
    if model_type == "qwen2_vl":
        model_type = model_type.replace('_', '')
    router = importlib.import_module(router_path)
    router_cls = getattr(router, f"{model_type.capitalize()}Router")
    router_ins = router_cls(
        model_name_or_path,
        config_dict,
        is_flash_causal_lm,
        load_tokenizer,
        max_position_embeddings,
        revision,
        tokenizer_path,
        trust_remote_code,
        enable_atb_torch)
    return router_ins