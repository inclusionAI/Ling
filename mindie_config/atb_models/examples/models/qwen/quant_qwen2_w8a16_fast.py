# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
import json
from typing import Dict, List, Optional

import torch
import safetensors
from safetensors.torch import save_file

from atb_llm.utils.file_utils import safe_open
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from examples.convert.convert_utils import copy_tokenizer_files, modify_config

DISABLE_NAMES = ["mlp.down_proj"]
WEIGHT_NAME = "quant_model_weight_w8a16.safetensors"
JSON_NAME = "quant_model_description_w8a16.json"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str
    )
    parser.add_argument(
        "save_path",
        type=str
    )
    return parser.parse_args()


def load_model(
        model_path: str,
        torch_dtype: str,
        device_type: str
):
    auto_args = {"device_map": "auto"} if device_type == "npu" else {}
    model = safe_get_model_from_pretrained(
        model_path,
        torch_dtype=getattr(torch, torch_dtype),
        trust_remote_code=False,
        **auto_args
    )
    return model if device_type == "npu" else model.cpu()


def quant_model(
        model_path: str,
        quant_save_path: str,
        torch_dtype: str,
        device_type: str,
        disable_names_list: Optional[List[str]] = None
) -> None:
    model = load_model(model_path, torch_dtype, device_type)
    config = Qwen2Config.from_pretrained(model_path)

    disable_names = []
    if disable_names_list:
        for layer in range(config.num_hidden_layers):
            for disable_name in disable_names_list:
                disable_names.append(f"model.layers.{layer}.{disable_name}")
    disable_names.append("lm_head")

    quant_config = QuantConfig(
        w_bit=8,                      # 权重量化位数
        a_bit=16,                     # 激活值量化位数
        disable_names=disable_names,  # 不做量化的层
        dev_type=device_type,         # 量化设备
        pr=1.0,                       # 量化正则百分比
        w_sym=True,                   # 对称/非对称量化，True为对称量化，False为非对称量化
        mm_tensor=False               # 权重量化粒度，True为per-tensor量化，False为per-channel量化（大模型场景建议False）
    )
    calibrator = Calibrator(
        model,
        quant_config,
        calib_data=None,    # W8A16量化无需校准
        disable_level="L0"  # 自动回退等级，根据精度损失程度增加不量化的层（L0~L5，L0为不回退，精度损失明显时可适当提升等级）
    )

    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(quant_save_path, save_type=["safe_tensor"])


def get_tensors(weight_path: str) -> Dict:
    tensors = {}
    with safetensors.safe_open(weight_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def fusion_weight(cpu_weight_path: str, npu_weight_path: str, save_path: str, disable_names_list: List[str]) -> None:
    cpu_tensors = get_tensors(os.path.join(cpu_weight_path, WEIGHT_NAME))
    npu_tensors = get_tensors(os.path.join(npu_weight_path, WEIGHT_NAME))

    tensors = {}
    for k, v in cpu_tensors.items():
        for disable_name in disable_names_list:
            if disable_name in k:
                if k.endswith("weight"):
                    tensors[k] = npu_tensors.get(k)
            else:
                tensors[k] = v

    os.makedirs(save_path, exist_ok=True)
    save_file(tensors, os.path.join(save_path, WEIGHT_NAME), metadata={"format": "pt"})


def fusion_json(cpu_weight_path: str, npu_weight_path: str, save_path: str, disable_names_list: List[str]) -> None:
    encoding = "utf-8"

    with safe_open(os.path.join(cpu_weight_path, JSON_NAME), mode="r", encoding=encoding) as cf:
        cpu_dict = json.load(cf)
    with safe_open(os.path.join(npu_weight_path, JSON_NAME), mode="r", encoding=encoding) as nf:
        npu_dict = json.load(nf)

    fusion_dict = {}
    for k, v in cpu_dict.items():
        for disable_name in disable_names_list:
            if disable_name in k:
                if k.endswith("weight"):
                    fusion_dict[k] = npu_dict.get(k)
            else:
                fusion_dict[k] = v

    os.makedirs(save_path, exist_ok=True)
    with safe_open(os.path.join(save_path, JSON_NAME), "w", encoding=encoding) as f:
        json.dump(fusion_dict, f, ensure_ascii=False, indent=2)


class Quantifier:
    def __init__(self, model_path: str, quant_save_path: str) -> None:
        self.model_path = model_path
        self.quant_save_path = quant_save_path
        self.cpu_quant_path = os.path.join(self.quant_save_path, "cpu")
        self.npu_quant_path = os.path.join(self.quant_save_path, "npu")

    def __call__(self, disable_names_list: List[str]) -> None:
        self.quant(disable_names_list)
        self.fusion(disable_names_list)

        modify_config(self.model_path, self.quant_save_path, torch.bfloat16, "w8a16")
        copy_tokenizer_files(self.model_path, self.quant_save_path)

    def quant(self, disable_names_list: List[str]) -> None:
        quant_model(
            self.model_path,
            self.cpu_quant_path,
            "float32",
            "cpu"
        )
        quant_model(
            self.model_path,
            self.npu_quant_path,
            "bfloat16",
            "npu",
            disable_names_list
        )

    def fusion(self, disable_names_list: List[str]) -> None:
        fusion_args = {
            "cpu_weight_path": self.cpu_quant_path,
            "npu_weight_path": self.npu_quant_path,
            "save_path": self.quant_save_path,
            "disable_names_list": disable_names_list
        }

        fusion_weight(**fusion_args)
        fusion_json(**fusion_args)


if __name__ == "__main__":
    args = parse_arguments()
    quantifier = Quantifier(args.model_path, args.save_path)
    quantifier(DISABLE_NAMES)