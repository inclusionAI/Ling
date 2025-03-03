# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import logging
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Union
from atb_llm.utils.env import ENV

import psutil


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
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = \
            self._info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


@dataclass
class CPUBinder:
    logger: logging.Logger = logging.getLogger()

    @staticmethod
    def _get_device_map_info() -> Dict[int, DeviceInfo]:
        device_map_info = {}
        device_map = \
            execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
        for line in device_map:
            device_info = DeviceInfo(line.strip())
            if isinstance(device_info.chip_logic_id, int):
                device_map_info[device_info.chip_logic_id] = device_info
        return device_map_info

    @staticmethod
    def _get_pcie_info(devices: List[int], keyword="PCIeBusInfo"):
        device_map_info = CPUBinder._get_device_map_info()
        device_pcie_tbl = {}
        for device in devices:
            device_info = device_map_info.get(device)
            if not device_info:
                raise RuntimeError("Can not get device info, binding cpu will skip.")
            pcie_info = \
                execute_command(["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}",
                                         "-c", f"{device_info.chip_id}"]).strip().split("\n")
            for _ in pcie_info:
                line = ''.join(_.split())
                if line.startswith(keyword):
                    device_pcie_tbl[device] = line[len(keyword) + 1:]
                    break

        return device_pcie_tbl

    @staticmethod
    def _get_numa_info(pcie_tbl, keyword="NUMAnode"):
        device_numa_tbl = {}  # key is device id, value is numa id
        numa_devices_tbl = {}  # key is numa id, value is device id list

        for device, pcie_no in pcie_tbl.items():
            numa_info = execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).strip().split("\n")
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

    @staticmethod
    def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
        cpu_idx_tbl = dict()
        numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
        cpu_info = execute_command(["lscpu"]).strip().split("\n")
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

    def bind_cpus(self, visible_devices_set: List[int] = None, rank_id: int = 0, ratio: float = 0.5):
        """
        可以用export CPU_BINDING_NUM设置每个进程绑的核数;如果不设置CPU_BINDING_NUM,
        会根据ratio(numa利用率)进行计算,如果有64个核，0.5表示用一半，用32个核, 平分给亲和在这个numa上的npu
        :param visible_devices:
        :param rank_id:
        :param ratio:
        :return:
        """

        if visible_devices_set is None:
            devices = [
                int(item.strip())
                for item in ENV.visible_devices.split(",")
                if item.isnumeric()
            ]
        else:
            devices = visible_devices_set

        # 获取npu和pcie的对应关系
        device_pcie_tbl = self._get_pcie_info(devices)
        # 根据pcie信息获取npu和numa的对应关系
        device_numa_tbl, numa_devices_tbl = self._get_numa_info(device_pcie_tbl)
        # 获取使用的numa对应的cpu核分配信息
        cpu_idx_tbl = self._get_cpu_info(list(numa_devices_tbl.keys()))

        # 当前rank的npu id
        cur_device = devices[rank_id]
        # 获取npu对应的numa id
        numa_id = device_numa_tbl.get(cur_device)

        # 获取共享该numa的npu信息
        shard_devices = numa_devices_tbl.get(numa_id)
        # 按照npu id进行排序
        shard_devices.sort()

        # 获取该numa上所有的cpu id信息
        all_cpus = cpu_idx_tbl[numa_id]
        info_msg = (f"rank_id: {rank_id}, device_id: {cur_device}, numa_id: {numa_id}, "
                    f"shard_devices: {shard_devices}, cpus: {all_cpus}")
        self.logger.info(info_msg)

        cpu_nums = len(all_cpus)
        # 计算给该共享numa的npu分配的核的个数
        cpu_binding_num = ENV.cpu_binding_num
        if cpu_binding_num is None:
            cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
        else:
            cpu_num_per_device = int(cpu_binding_num)
            if len(shard_devices) * cpu_num_per_device > cpu_nums:
                raise Exception(
                    f"Cpu num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                    f"please decrease the value of CPU_BINDING_NUM!")

        # 获取该npu的下标信息
        idx = shard_devices.index(cur_device)
        # 给该npu分配要绑定的cpu id
        binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]

        # cpu bind
        p = psutil.Process()
        p.cpu_affinity(binding_cpus)
        new_affinity = p.cpu_affinity()
        info_msg = f"process {p.pid}, new_affinity is {new_affinity}, cpu count {cpu_num_per_device}"
        self.logger.info(info_msg)
