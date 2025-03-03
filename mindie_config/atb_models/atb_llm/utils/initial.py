# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
from enum import Enum
from dataclasses import dataclass

import torch
import torch_npu

from .cpu_binding import execute_command
from .env import ENV


class Topo(str, Enum):
    pcie = "pcie"
    hccs = "hccs"


class CommunicationLibrary(str, Enum):
    hccl = "hccl"
    lccl = "lccl"


@dataclass
class NPUSocInfo:
    soc_name: str = ""
    soc_version: int = -1
    need_nz: bool = False
    matmul_nd_nz: bool = False

    def __post_init__(self):
        self.soc_version = torch_npu._C._npu_get_soc_version()
        if self.soc_version in (100, 101, 102, 103, 104, 200, 201, 202, 203):
            self.need_nz = True
            
    @property
    def communication_backend(self):
        return CommunicationLibrary.lccl \
        if self.is_support_lccl() and not ENV.hccl_enable else CommunicationLibrary.hccl
    
    def is_support_lccl(self):
        npu_smi_info = execute_command(["npu-smi", "info", "-t", "topo"])
        legend_index = npu_smi_info.find("Legend")
        rank_table_file = ENV.rank_table_file
        npu_vm_support_hccs = ENV.npu_vm_support_hccs or Topo.hccs in npu_smi_info[:legend_index].lower()
        return not self.need_nz and npu_vm_support_hccs and not rank_table_file
        

def load_atb_speed():
    lib_path = os.path.join(ENV.atb_speed_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)
    sys.path.append(os.path.join(ENV.atb_speed_home_path, 'lib'))


def is_lcoc_enable(need_nz):
    lcoc_enable = ENV.lcoc_enable and (not need_nz)
    return lcoc_enable


def check_profiling_level():
    profiler_level = torch_npu.profiler.ProfilerLevel
    if not hasattr(profiler_level, ENV.profiling_level):
        raise ValueError(f"target_level: {ENV.profiling_level} is not implemented"
                         f" in torch_npu.profiler.ProfilerLevel")