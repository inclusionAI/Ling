# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import math
import json
import re
from enum import Enum
from typing import List, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch_npu

from safetensors import safe_open

from atb_llm.utils.weights import Weights
from .hub import weight_files
from .log import print_log, logger
from .quantize.pack_type import TransposeType
from .layers.linear.lora import Lora
from ..models.base.config import LoraConfig
from .layers.linear.linear_utils import LinearUtils
from .file_utils import standardize_path, check_file_safety, check_path_permission
from .file_utils import safe_open as file_utils_safe_open
from .argument_utils import MAX_KEY_LENGTH


class AdapterWeightLoader(Weights):
    def __init__(self, lora_adapter_path, device, dtype, process_group):
        super().__init__(lora_adapter_path, device, dtype, process_group)

    def update_weights_path(self, model_name_or_path):
        model_name_or_path = standardize_path(model_name_or_path, check_link=False)
        check_path_permission(model_name_or_path)
        self.filenames = weight_files(model_name_or_path)
        routing = {}
        for filename in self.filenames:
            filename = standardize_path(str(filename), check_link=False)
            check_path_permission(filename)
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        print_log(
                            self.process_group.rank(),
                            logger.error,
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}",
                            need_filter=True
                        )
                        raise AssertionError
                    routing[k] = filename
        self.routing = routing

    def get_lora_tensor(self, tensor_name: str, is_lora_a: bool,
        padding_shape: tuple, tensor_parallel_dim, align_size) -> torch.Tensor:
        try:
            # QKV和Gate Up Linear切n维，切Lora B；Dense和Down层切k维，切Lora A
            is_qkv_lora_b = (tensor_parallel_dim == 0 and not is_lora_a)
            is_gate_up_lora_a = (tensor_parallel_dim == 1 and is_lora_a)
            if is_qkv_lora_b or is_gate_up_lora_a:
                lora_tensor = self.get_sharded(tensor_name, dim=tensor_parallel_dim, gqa_size=align_size)
            else:
                lora_tensor = self.get_tensor(tensor_name)
        except AssertionError as _:
            print_log(self.process_group.rank(), logger.info,
                f"tensor name {tensor_name} not found, using all zero tensor of shape "
                f"{padding_shape} instead", need_filter=True)
            lora_tensor = torch.zeros(padding_shape)
        return self.correct_tensor_dtype(lora_tensor, tensor_name)


BASE_ADAPTER_NAME = "base"
SORTED_ADAPTER_NAME = "sorted"
LORA_CONFIG_NAME = "adapter_config.json"


@dataclass
class AdapterInfo:
    idx: int
    adapter_path: str
    config: LoraConfig | None = None
    weights: List | None = None


class AdapterIdsType(str, Enum):
    SINGLE = "single"
    MIXED = "mixed"
    SORTED = "sorted"


@dataclass
class AdapterIdsRecord:
    record_type: AdapterIdsType
    adapter_ids: List[str]


def format_lora_a_key(base_weight_prefix):
    return f"base_model.model.{base_weight_prefix}.lora_A.weight"


def format_lora_b_key(base_weight_prefix):
    return f"base_model.model.{base_weight_prefix}.lora_B.weight"


class AdapterManager:
    def __init__(self, weights: Weights):
        self.adapter_info_registry = {}  # keep track of adapter information
        self.previous_adapter_ids = None  # adapter ids in the last forward call
        self.base_model = None
        self.lm_head_prefix = "lm_head"
        self.format_lora_a_key = format_lora_a_key
        self.format_lora_b_key = format_lora_b_key

        self.base_weights_loader = weights
        self.lora_weights_loader = None
        self.dtype = weights.dtype
        self.process_group = weights.process_group
        self.place_holder = torch.zeros([1], dtype=self.dtype, device=weights.device)

        # 若仅启用单一Lora，则打开此开关
        self.enable_single_adapter_only = False

    @staticmethod
    def post_process_lora_a_weight(lora_weight: List[torch.Tensor]) -> torch.Tensor:
        if len(lora_weight) == 1:
            lora_weight = lora_weight[0]
        else:
            lora_weight = torch.cat(lora_weight, 0)
        return lora_weight

    @staticmethod
    def padding_tensor(padding_axis, padding_size, tensor):
        padding_shape = [0, 0] * len(tensor.shape)
        padding_shape[-(padding_axis * 2 + 1)] = \
            (padding_size - tensor.shape[padding_axis] % padding_size) % padding_size
        return F.pad(tensor, padding_shape)
    
    @staticmethod
    def get_last_two_prefix(prefix: str) -> str:
        """
        if the prefix is `model.layer.0.attention.qkv`,
        it will return string `attention.qkv`
        """
        match = re.search(r'([^\.]+\.[^\.]+)$', prefix)
        if match:
            return match.group(1) # The first parenthesized subgroup.
        return ""

    def get_alpha(self, lora_config: LoraConfig, prefix: str) -> int:
        lora_alpha = lora_config.lora_alpha
        if lora_config.alpha_pattern is not None:
            alpha_prefix = self.get_last_two_prefix(prefix)
            lora_alpha = lora_config.alpha_pattern.get(alpha_prefix, lora_alpha)
        return lora_alpha

    def get_r(self, lora_config: LoraConfig, prefix: str) -> int:
        lora_r = lora_config.r
        if lora_config.rank_pattern is not None:
            r_prefix = self.get_last_two_prefix(prefix)
            lora_r = lora_config.rank_pattern.get(r_prefix, lora_r)
        return lora_r

    def get_scaling(self, lora_config: LoraConfig, prefix: str) -> int:
        lora_alpha = self.get_alpha(lora_config, prefix)
        lora_r = self.get_r(lora_config, prefix)
        if lora_config.use_rslora:
            scaling = lora_alpha / math.sqrt(lora_r)
        else:
            scaling = lora_alpha / lora_r
        return scaling
    
    def get_base_weight_shape(self, linear_module, tensor_name=None) -> tuple:
        if linear_module.trans_flag == TransposeType.INVALID:
            raise ValueError(f"trans_flag of linear_module with prefixes {linear_module.prefixes} is invalid")
        # 默认获取linear_module中的权重shape，若提供tensor_name，则直接从原始权重中获取tensor_name对应的权重shape
        weight_shape = linear_module.weight.shape
        if len(weight_shape) <= 1:
            return 1, 1
        if tensor_name is not None:
            weight_shape = self.base_weights_loader.get_sharded(
                tensor_name, dim=linear_module.tensor_parallel_dim, gqa_size=linear_module.align_size).shape
            return weight_shape
        if linear_module.trans_flag == TransposeType.TRANSPOSE:
            n, k = weight_shape
        else:
            k, n = weight_shape
        return n, k

    def weight_format_cast(self, tensor):
        if not self.base_model.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    def preload_adapter(self, lora_adapter: dict) -> None:
        # 加载adapter_info中的权重；先在CPU上进行加载，后续随整个model的权重一起搬运到npu上
        for adapter_id, adapter_path in lora_adapter.items():
            if len(adapter_id) <= 0 or len(adapter_id) > MAX_KEY_LENGTH:
                raise ValueError(f"Lora names should be no more than {MAX_KEY_LENGTH} characters long "
                                 "and at least one character long.")
            self.load_adapter(adapter_id, adapter_path, "cpu")
        
        if not self.enable_single_adapter_only:
            # 加载仅调用base model下的dummy权重
            self.load_dummy_adapter()

        # 对已加载的权重增加padding
        for linear_module in self._find_lora_module(self.base_model):
            lora_module = linear_module.lora
            self.add_padding(lora_module)
        
        if not self.enable_single_adapter_only:
            # 加载Sorted Concate权重
            self.concate_sorted_adapter()
    
    def prepare_adapter_weights(self):
        # 提前遍历所有linear module，获取adapter权重
        for adapter_id, adapter_info in self.adapter_info_registry.items():
            adapter_info.weights = self.get_single_adapter([adapter_id])
            # NZ格式转换
            if self.base_model.soc_info.need_nz:
                for i, tensor in enumerate(adapter_info.weights):
                    adapter_info.weights[i] = self.weight_format_cast(tensor)

    def load_adapter(self, adapter_id: None | str, adapter_path: str, device: str) -> None:
        # adapter_id预处理
        if adapter_id is None:
            adapter_id = BASE_ADAPTER_NAME
        if adapter_id == BASE_ADAPTER_NAME or adapter_id == SORTED_ADAPTER_NAME:
            raise NameError(f"adapter_id `{BASE_ADAPTER_NAME}` and `{SORTED_ADAPTER_NAME}` are reserved. "
                            f"Please change another adapter id.")

        # 检查是否已经加载
        adapter_info = self.adapter_info_registry.get(adapter_id)
        if adapter_info is not None:
            # 如果adapter_id已经加载过，就直接返回
            return

        # 第一次使用lora_weights_loader时需要初始化，后续直接更新权重路径即可
        if self.lora_weights_loader is None:
            self.lora_weights_loader = AdapterWeightLoader(adapter_path, device="cpu", dtype=self.dtype,
                                                      process_group=self.process_group)
        else:
            self.lora_weights_loader.update_weights_path(adapter_path)
        
        # 加载lora config
        adapter_config_path = standardize_path(os.path.join(adapter_path, LORA_CONFIG_NAME), check_link=True)
        check_file_safety(adapter_config_path)
        lora_config_in_dict = None
        with file_utils_safe_open(adapter_config_path, mode="r", encoding="utf-8") as f:
            lora_config_in_dict = json.load(f)
        lora_config = LoraConfig.from_dict(lora_config_in_dict)

        # 遍历base_model中的module，找到Lora module，加载权重
        for linear_module in self._find_lora_module(self.base_model):
            # 往moduleDict中保存adapter_id对应的weight tensor
            lora_a_weight = self.load_lora_weight(linear_module, is_lora_a=True)
            lora_a_weight = self.post_process_lora_a_weight(lora_a_weight)  # [r, k]
            lora_a_weight.to(device)
            lora_b_weight = self.load_lora_weight(linear_module, is_lora_a=False)
            lora_b_weight = self.post_process_lora_b_weight(
                lora_b_weight, lora_config, linear_module.prefixes)  # [r, n]
            lora_b_weight.to(device)
            if not hasattr(linear_module, "lora"):
                setattr(linear_module, "lora", nn.ModuleDict())
            r_list = []
            alpha_list = []
            for prefix in linear_module.prefixes:
                r_list.append(self.get_r(lora_config, prefix))
                alpha_list.append(self.get_alpha(lora_config, prefix))
            linear_module.lora[adapter_id] = Lora(lora_a_weight, lora_b_weight, r=r_list, alpha=alpha_list)

        # 注册adapter
        self.adapter_info_registry[adapter_id] = AdapterInfo(
            idx=len(self.adapter_info_registry), adapter_path=adapter_path, config=lora_config)

    def load_dummy_adapter(self):
        for linear_module in self._find_lora_module(self.base_model):
            # 获取原始权重的shape信息
            n, k = self.get_base_weight_shape(linear_module)
            if not hasattr(linear_module, "lora"):
                setattr(linear_module, "lora", nn.ModuleDict())
            linear_module.lora[BASE_ADAPTER_NAME] = \
                Lora(torch.zeros([1, k], dtype=self.dtype), torch.zeros([1, n], dtype=self.dtype),
                     r=1, alpha=1)
        self.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.adapter_info_registry), adapter_path="",
            config=LoraConfig(r=1, lora_alpha=1, use_rslora=False))

    def concate_sorted_adapter(self):
        for linear_module in self._find_lora_module(self.base_model):
            lora_module = linear_module.lora
            # 多个adapter权重合并
            linear_lora_a_adapter_weights = []
            linear_lora_b_adapter_weights = []
            sorted_adatper_id = sorted(self.adapter_info_registry.keys(),
                key=lambda adapter_id: self.adapter_info_registry.get(adapter_id).idx)
            for adapter_id in sorted_adatper_id:
                linear_lora_a_adapter_weights.append(lora_module[adapter_id].lora_a.data.unsqueeze(0))
                linear_lora_b_adapter_weights.append(lora_module[adapter_id].lora_b.data.unsqueeze(0))
            # padding后shape为[# of adapters, max(ri), k]
            padded_lora_a = torch.cat(linear_lora_a_adapter_weights, dim=0)
            # shape为[# of adapters, max(ri), n]
            padded_lora_b = torch.cat(linear_lora_b_adapter_weights, dim=0)
            if not hasattr(linear_module, "lora"):
                setattr(linear_module, "lora", nn.ModuleDict())
            linear_module.lora[SORTED_ADAPTER_NAME] = Lora(padded_lora_a, padded_lora_b, r=1, alpha=1)
        self.adapter_info_registry[SORTED_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.adapter_info_registry), adapter_path="",
            config=LoraConfig(r=1, lora_alpha=1, use_rslora=False))

    def add_padding(self, lora_module: nn.ModuleDict):
        max_r = -1
        for adapter in lora_module.keys():
            max_r = max(max_r, lora_module[adapter].lora_a.data.shape[0])
        for adapter in lora_module.keys():
            # 不同硬件r维度对齐策略不同
            padding_size = (max_r // 16 + 1) * 16 if self.base_model.soc_info.need_nz else (max_r // 64 + 1) * 64
            lora_module[adapter].lora_a.data = self.padding_tensor(
                0, padding_size, lora_module[adapter].lora_a.data)  # [r, k]
            lora_module[adapter].lora_b.data = self.padding_tensor(
                0, padding_size, lora_module[adapter].lora_b.data)  # [r, n]

    def preprocess_adatper_ids(
        self, adapter_ids: None | List[str | None],
        model_name=None, model_id=None) -> List[str]:
        if adapter_ids is None:
            adapter_ids = [None]
        # 校验、adapter_id转换
        effective_adapter_ids = []
        for adapter_id in adapter_ids:
            if adapter_id == model_name or adapter_id == model_id:
                effective_adapter_ids.append(BASE_ADAPTER_NAME)
            elif adapter_id and adapter_id not in self.adapter_info_registry.keys():
                print_log(self.process_group.rank(), logger.warn,
                    f"Adapter `{adapter_id}` is not found, using base model instead",
                    need_filter=True)
                effective_adapter_ids.append(BASE_ADAPTER_NAME)
            elif adapter_id is None:
                effective_adapter_ids.append(BASE_ADAPTER_NAME)
            else:
                effective_adapter_ids.append(adapter_id)
        # adapter归一
        unique_adapter_ids = set(effective_adapter_ids)
        if len(unique_adapter_ids) == 1:
            effective_adapter_ids = list(unique_adapter_ids)
        return effective_adapter_ids

    def update_adapter(self, adapter_ids: None | List[str | None]) -> bool:
        # 输入的adapter_ids已经过preprocess_adatper_ids处理
        if len(adapter_ids) == 1:
            current_adapter_ids_cache = AdapterIdsRecord(AdapterIdsType.SINGLE, adapter_ids)
        elif self.check_adapter_ids_is_sorted(adapter_ids):
            current_adapter_ids_cache = AdapterIdsRecord(AdapterIdsType.SORTED, adapter_ids)
        else:
            current_adapter_ids_cache = AdapterIdsRecord(AdapterIdsType.MIXED, adapter_ids)

        if self.previous_adapter_ids is None:
            self.previous_adapter_ids = current_adapter_ids_cache
            return True
        if self.previous_adapter_ids == current_adapter_ids_cache:
            return False
        if self.previous_adapter_ids.record_type == current_adapter_ids_cache.record_type == AdapterIdsType.SORTED:
            return False
        self.previous_adapter_ids = current_adapter_ids_cache
        return True

    def check_adapter_ids_is_sorted(self, adapter_ids: None | List[str | None]) -> bool:
        if adapter_ids is None:
            adapter_ids = [None]
        adapter_idx = -1
        for adapter_id in adapter_ids:
            if adapter_id is None or adapter_id not in self.adapter_info_registry.keys():
                print_log(self.process_group.rank(), logger.warn,
                    f"Adapter {adapter_id} is not found, using base model instead",
                    need_filter=True)
                cur_adapter_idx = self.adapter_info_registry.get(BASE_ADAPTER_NAME).idx
            else:
                cur_adapter_idx = self.adapter_info_registry.get(adapter_id).idx
            if cur_adapter_idx < adapter_idx:
                return False
            adapter_idx = cur_adapter_idx
        return True

    def sort_adapter_ids(self, adapter_ids: List[str]) -> Union[List[int], List[int]]:
        # 输入的adapter_ids已经过preprocess_adatper_ids处理
        sorted_adapter_ids_by_idx = sorted(enumerate(adapter_ids),
            key=lambda idx_adapter_id: self.adapter_info_registry.get(idx_adapter_id[1]).idx)
        sorted_adapter_idx = [idx for idx, _ in sorted_adapter_ids_by_idx]
        revert_adapter_idx = [idx for idx, _ in sorted(enumerate(sorted_adapter_idx), key=lambda group: group[1])]
        return sorted_adapter_idx, revert_adapter_idx

    def get_adapters(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        if self.previous_adapter_ids.record_type == AdapterIdsType.SINGLE:
            adapter_weights = self.get_single_adapter(adapter_ids)
        elif self.previous_adapter_ids.record_type == AdapterIdsType.SORTED:
            adapter_weights = self.get_single_adapter([SORTED_ADAPTER_NAME])
        else:
            adapter_weights = self.get_mixed_adapter(adapter_ids)
            # NZ格式转换
            if self.base_model.soc_info.need_nz:
                for i, tensor in enumerate(adapter_weights):
                    adapter_weights[i] = self.weight_format_cast(tensor)
        return adapter_weights

    def get_single_adapter(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        adapter_info = self.adapter_info_registry.get(adapter_ids[0])
        if adapter_info is not None and adapter_info.weights is not None:
            return adapter_info.weights
        adapter_weights = []
        # 遍历base_model中的module，找到Lora module
        for linear_module in self._find_lora_module(self.base_model):
            lora_module = linear_module.lora
            adapter_weights.append(lora_module[adapter_ids[0]].lora_a.data)
            adapter_weights.append(lora_module[adapter_ids[0]].lora_b.data)
            # 检测是否为pack场景（num_linear_before_pack数量是否大于1），需要补placeholder
            for _ in range(linear_module.num_linear_before_pack - 1):
                adapter_weights.extend([self.place_holder, self.place_holder])  # lora_a, lora_b
        return adapter_weights

    def get_mixed_adapter(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        adapter_weights = []
        # 遍历base_model中的module，找到Lora module
        for linear_module in self._find_lora_module(self.base_model):
            lora_module = linear_module.lora
            # 多个adapter权重合并
            linear_lora_a_adapter_weights = []
            linear_lora_b_adapter_weights = []
            for adapter_id in adapter_ids:
                linear_lora_a_adapter_weights.append(lora_module[adapter_id].lora_a.data.unsqueeze(0))
                linear_lora_b_adapter_weights.append(lora_module[adapter_id].lora_b.data.unsqueeze(0))
            # padding后shape为[bs, max(ri), k]，算子需要[bs, k, max(ri)]
            padded_lora_a = torch.cat(linear_lora_a_adapter_weights, dim=0)
            # shape为[bs, max(ri), n]
            padded_lora_b = torch.cat(linear_lora_b_adapter_weights, dim=0)
            adapter_weights.append(padded_lora_a)
            adapter_weights.append(padded_lora_b)
            # 检测是否为pack场景（num_linear_before_pack数量是否大于1），需要补placeholder
            for _ in range(linear_module.num_linear_before_pack - 1):
                adapter_weights.extend([self.place_holder, self.place_holder])  # lora_a, lora_b
        return adapter_weights

    def load_lora_weight(self, lora_parent_module: LinearUtils, is_lora_a: bool) -> List[torch.Tensor]:
        tmp_lora_weight = []
        for prefix in lora_parent_module.prefixes:
            if prefix is None:
                # 原生模型的QKV和gateup就是合并的场景会进入此分支
                continue
            if is_lora_a:
                lora_prefix = self.format_lora_a_key(prefix)
            else:
                lora_prefix = self.format_lora_b_key(prefix)
            # 获取原始权重的shape信息
            n, k = self.get_base_weight_shape(lora_parent_module, f"{prefix}.weight")
            # 如果lora权重不存在，则lora_A权重创建shape=(1, k)的全零tensor，lora_B权重创建shape=(n, 1)的全零tensor
            if is_lora_a:
                padding_shape = (1, k)
            else:
                padding_shape = (n, 1)
            weight_tensor = self.lora_weights_loader.get_lora_tensor(
                lora_prefix, is_lora_a, padding_shape,
                tensor_parallel_dim=lora_parent_module.tensor_parallel_dim,
                align_size=lora_parent_module.align_size
            )
            tmp_lora_weight.append(weight_tensor)
        return tmp_lora_weight

    def post_process_lora_b_weight(self, lora_weight: List[torch.Tensor],
                                   lora_config: LoraConfig, prefixes: List[str]) -> torch.Tensor:
        if len(lora_weight) == 1:
            lora_weight = lora_weight[0] * self.get_scaling(lora_config, prefixes[0])
        else:
            scaled_lora_weights = []
            for weight_tensor, prefix in zip(lora_weight, prefixes):
                scaled_lora_weights.append(weight_tensor * self.get_scaling(lora_config, prefix))
            lora_weight = torch.block_diag(*scaled_lora_weights)
        # original Shape: lora_A [r, k] lora_B [n, r]
        # GroupedMatMulOperation and LinearOperation need lora_A of shape [r, k] and lora_B of shape [r, n]
        # pad_sequence do padding on the first dimension, we want to pad on r
        # so transpose lora_B here
        lora_weight = lora_weight.T.contiguous()
        return lora_weight

    def _find_lora_module(self, module):
        for name, child_module in module.named_children():
            # recursive查找Linear
            if self.lm_head_prefix in name:  # 不加载LmHead中的Lora权重
                continue
            if isinstance(child_module, nn.Module):
                yield from self._find_lora_module(child_module)
            elif isinstance(child_module, nn.ModuleList):
                for single_module in child_module:
                    yield from self._find_lora_module(single_module)
            elif isinstance(child_module, nn.ModuleDict):
                for single_module in child_module.values():
                    yield from self._find_lora_module(single_module)
            if isinstance(child_module, LinearUtils):
                yield child_module
