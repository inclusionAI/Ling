# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
import subprocess
from typing import List, Dict, Union
import torch_npu

import psutil

from .env import ENV
from .log import logger


def execute_command(cmd_list):
    with subprocess.Popen(cmd_list,
                          shell=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        out, err = p.communicate(timeout=1000)
    res = out.decode()
    return res


@dataclass
class DeviceInfo:
    _info_line: str = ""
    npu_id: int = 0
    chip_id: int = 0
    chip_logic_id: Union[int, str] = 0
    chip_name: str = ""

    def __post_init__(self):
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = self._info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


class NpuHbmInfo:
    visible_npu_ids: List = None
    hbm_capacity: int = None
    hbm_usage: int = None

    @classmethod
    def set_visible_devices(cls, world_size):
        if cls.visible_npu_ids:
            return
        if ENV.visible_devices is None:
            devices = sorted(list(_get_device_map_info().keys()))  # 通过npu-smi info -m指令获取到所有的chip logic id
        else:
            devices = ENV.visible_devices
        device_map_info = _get_device_map_info()
        npu_ids = []
        for device in devices:
            device_info = device_map_info.get(device)
            npu_ids.append(device_info.npu_id)
        cls.visible_npu_ids = npu_ids

    @classmethod
    def get_hbm_capacity(cls, rank, world_size, need_nz):
        soc_version = torch_npu._C._npu_get_soc_version()
        if cls.hbm_capacity:
            return cls.hbm_capacity
        if not cls.visible_npu_ids:
            cls.set_visible_devices(world_size)
        npu_id = cls.visible_npu_ids[rank]
        memory_info = execute_command(["npu-smi", "info", "-i", f"{npu_id}", "-t", "memory"]).split("\n")[1:]
        if soc_version == 240:
            hbm_capacity_key = 'Capacity(MB)'
        elif not need_nz:
            hbm_capacity_key = 'HBM Capacity(MB)'
        else:
            hbm_capacity_key = 'DDR Capacity(MB)'
        for line in memory_info:
            try:
                key, value = line.strip().split(':', 2)
                if key.strip() == hbm_capacity_key:
                    cls.hbm_capacity = int(value.strip()) * 1024 * 1024
                    return cls.hbm_capacity
            except ValueError:
                pass
        raise ValueError('not found valid hbm capactiy')

    @classmethod
    def get_hbm_usage(cls, rank, world_size, need_nz):
        if cls.hbm_usage:
            return cls.hbm_usage
        if not cls.visible_npu_ids:
            cls.set_visible_devices(world_size)
        npu_id = cls.visible_npu_ids[rank]
        usage_info = execute_command(["npu-smi", "info", "-i", f"{npu_id}", "-t", "usages"]).split("\n")[1:]
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version == 240:
            hbm_capacity_key = 'Memory Usage Rate(%)'
        elif not need_nz:
            hbm_capacity_key = 'HBM Usage Rate(%)'
        else:
            hbm_capacity_key = 'DDR Usage Rate(%)'
        for line in usage_info:
            try:
                key, value = line.strip().split(':', 2)
                if key.strip() == hbm_capacity_key:
                    hbm_usage = (float(value.strip()) + 1) / 100
                    return hbm_usage
            except ValueError:
                pass
        raise ValueError('not found valid hbm usage')


def _get_device_map_info() -> Dict[int, DeviceInfo]:
    device_map_info = {}
    device_map = execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
    for line in device_map:
        device_info = DeviceInfo(line.strip())
        if isinstance(device_info.chip_logic_id, int):
            device_map_info[device_info.chip_logic_id] = device_info
    return device_map_info


def _get_pcie_info(devices: List[int], keyword="PCIeBusInfo"):
    device_map_info = _get_device_map_info()
    device_pcie_tbl = {}
    for device in devices:
        device_info = device_map_info.get(device)
        if not device_info:
            raise RuntimeError("Can not get device info, you can use BIND_CPU=0 to skip.")
        pcie_info = execute_command(["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}",
                                     "-c", f"{device_info.chip_id}"]).strip().split("\n")
        for _ in pcie_info:
            line = ''.join(_.split())  # 不同硬件的输出格式不同（PCIe Bus Info 或 PCIeBusInfo），在此处做统一
            if line.startswith(keyword):
                device_pcie_tbl[device] = line[len(keyword) + 1:]
                break

    return device_pcie_tbl


def _get_numa_info(pcie_tbl, keyword="NUMAnode"):
    device_numa_tbl = dict()  # key is device id, value is numa id
    numa_devices_tbl = dict()  # key is numa id, value is device id list

    for device, pcie_no in pcie_tbl.items():
        numa_info = execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).split("\n")
        for _ in numa_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1:])
                device_numa_tbl[device] = numa_id

                devices = numa_devices_tbl.get(numa_id, None)
                if devices is None:
                    numa_devices_tbl[numa_id] = list()

                numa_devices_tbl[numa_id].append(device)
                break

    return device_numa_tbl, numa_devices_tbl


def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = execute_command(["lscpu"]).split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")

                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

            numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


# 可以用export CPU_BINDING_NUM设置每个进程绑的核数;如果不设置CPU_BINDING_NUM,
# 会根据ratio(numa利用率)进行计算,如果有64个核，0.5表示用一半，用32个核, 平分给亲和在这个numa上的npu
def bind_cpus(world_size, rank_id, ratio=0.5):
    visible_devices = ENV.visible_devices

    if visible_devices is None:
        devices = sorted(list(_get_device_map_info().keys()))  # 通过npu-smi info -m指令获取到所有的chip logic id
    else:
        devices = ENV.visible_devices

    # 获取npu和pcie的对应关系
    device_pcie_tbl = _get_pcie_info(devices)
    # 根据pcie信息获取npu和numa的对应关系
    device_numa_tbl, numa_devices_tbl = _get_numa_info(device_pcie_tbl)
    # 获取使用的numa对应的cpu核分配信息
    cpu_idx_tbl = _get_cpu_info(list(numa_devices_tbl.keys()))

    # 当前rank的npu id
    cur_device = devices[rank_id]
    # 获取npu对应的numa id
    numa_id = device_numa_tbl.get(cur_device)

    # 获取共享该numa的npu信息
    shard_devices = numa_devices_tbl.get(numa_id)
    # 按照npu id进行排序
    shard_devices.sort()

    # 获取该numa上所有的cpu id信息
    all_cpus = cpu_idx_tbl.get(numa_id)
    logger.info(
        f"rank_id: {rank_id}, device_id: {cur_device}, "
        f"numa_id: {numa_id}, shard_devices: {shard_devices}, cpus: {all_cpus}")

    cpu_nums = len(all_cpus)
    # 计算给该共享numa的npu分配的核的个数
    if ENV.cpu_binding_num is None:
        cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
    else:
        cpu_num_per_device = int(ENV.cpu_binding_num)
        if len(shard_devices) * cpu_num_per_device > cpu_nums:
            raise RuntimeError(
                f"Cpu num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                f"please decrease the value of CPU_BINDING_NUM!")
        if cpu_num_per_device < 0:
            raise ValueError("CPU_BINDING_NUM should not be less than 0.")

    # 获取该npu的下标信息
    idx = shard_devices.index(cur_device)
    # 给该npu分配要绑定的cpu id
    binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]

    # cpu bind
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    logger.info(f"process {p.pid}, new_affinity is {new_affinity}, cpu count {cpu_num_per_device}")
