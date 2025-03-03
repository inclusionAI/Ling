# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import array
from collections import OrderedDict
import logging
import sys
import pandas as pd
import numpy as np
import torch
from atb_llm.utils.file_utils import safe_open, standardize_path, check_file_safety

np.seterr(divide="ignore", invalid="ignore")  # ignore RuntimeWarning: divide by zero encountered in divide
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype, self.format, self.dims = 0, 0, []
        self.dtype_dict = {
            0: torch.float32,
            1: torch.float16,
            2: torch.int8,
            3: torch.int32,
            9: torch.int64,
            12: torch.bool,
            27: torch.bfloat16
        }

        self.attr_version = "$Version"
        self.attr_end = "$End"
        self.attr_object_length = "$Object.Length"
        self.attr_object_count = "$Object.Count"
        self.attr_object_prefix = "$Object."

        self.__parse_bin_file()

    def get_data(self):
        if self.dtype not in self.dtype_dict:
            logging.error("error, unsupport dtype: %s", self.dtype)
            pass
        dtype = self.dtype_dict.get(self.dtype)
        info_message = "Load from bin file, dtype: {}, shape: {}".format(dtype, self.dims)
        logging.info(info_message)
        tensor = torch.frombuffer(array.array('b', self.obj_buffer), dtype=dtype)
        tensor = tensor.view(self.dims)
        return tensor

    def __parse_bin_file(self):
        with safe_open(self.file_path, "rb") as fd:
            file_data = fd.read()

            begin_offset = 0
            for i, byte in enumerate(file_data):
                if byte == ord("\n"):
                    line = file_data[begin_offset: i].decode("utf-8")
                    begin_offset = i + 1
                    fields = line.split("=")
                    attr_name = fields[0]
                    attr_value = fields[1]
                    if attr_name == self.attr_end:
                        self.obj_buffer = file_data[i + 1 :]
                        break
                    elif attr_name.startswith("$"):
                        self.__parse_system_atrr(attr_name, attr_value)
                    else:
                        self.__parse_user_attr(attr_name, attr_value)
                        pass

    def __parse_system_atrr(self, attr_name, attr_value):
        if attr_name == self.attr_object_length:
            self.obj_len = int(attr_value)
        elif attr_name == self.attr_object_prefix:
            pass

    def __parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            self.dims = [int(dim) for dim in self.dims]


def cosine(data1, data2):
    return torch.cosine_similarity(data1, data2, dim=0)


def euclidean(data1, data2):
    return ((data1 - data2) ** 2).sum() ** 0.5


def root_mean_square(data1, data2):
    return ((data1 - data2) ** 2).mean() ** 0.5


def absolute(data1, data2):
    return (data1 - data2).abs()


def relative(data1, data2):
    data1_abs = data1.abs()
    return torch.where(data1_abs > 1e-12, (data1 - data2).abs() / data1_abs, torch.zeros_like(data1))


def kl_divergence(data1, data2):
    if data1.min() == data1.max() or data2.min() == data2.max():
        raise ValueError("Input data cannot have the same minumum and maximun values.")
    norm_data1 = (data1 - data1.min()) / (data1.max() - data1.min())
    norm_data2 = (data2 - data2.min()) / (data2.max() - data2.min())

    norm_data1 /= norm_data1.sum()
    norm_data2 /= norm_data2.sum()
    valid_cond = torch.logical_and(norm_data1 > 1e-12, norm_data2 > 1e-12)
    return (norm_data1 * torch.where(valid_cond, (norm_data1 / norm_data2).log(), torch.zeros_like(norm_data1))).sum()


def compare(data1, data2):
    result = OrderedDict()
    result["cosine"] = cosine(data1, data2).item()
    euclidean_result = euclidean(data1, data2).item()
    result["absolute_euclidean"] = euclidean_result
    if (data1**2).sum() == 0:
        raise ValueError("Input data cannot be zeros.")
    result["relative_euclidean"] = euclidean_result / ((data1**2).sum().item() ** 0.5)
    result["root_mean_square"] = root_mean_square(data1, data2)  # euclidean_result / (data1.shape[0] ** 0.5)
    result["kl_divergence"] = kl_divergence(data1, data2)
    result[""] = ""  # Empty line

    absolute_result = absolute(data1, data2)
    result["absolute_max"] = absolute_result.max().item()
    result["absolute_mean"] = absolute_result.mean().item()

    relative_result = relative(data1, data2)
    result["relative_max"] = relative_result.max().item()
    result["relative_mean"] = relative_result.mean().item()
    result = {kk: "" if vv == "" else "{:.6g}".format(vv) for kk, vv in result.items()}

    result[" "] = ""  # Empty line
    round_format = "{:.4f}%"
    if absolute_result.shape[0] != 0:
        result["absolute < 0.1"] = round_format.format(
            100 * (absolute_result < 0.1).sum() / absolute_result.shape[0])
        result["absolute < 0.01"] = round_format.format(
            100 * (absolute_result < 0.01).sum() / absolute_result.shape[0])
        result["absolute < 0.001"] = round_format.format(
            100 * (absolute_result < 0.001).sum() / absolute_result.shape[0])
        result["absolute < 0.0001"] = round_format.format(
            100 * (absolute_result < 0.0001).sum() / absolute_result.shape[0])
        
    result["  "] = ""  # Empty line
    round_format = "{:.4f}%"
    if relative_result.shape[0] != 0:
        result["relative < 0.1"] = round_format.format(
            100 * (relative_result < 0.1).sum() / relative_result.shape[0])
        result["relative < 0.01"] = round_format.format(
            100 * (relative_result < 0.01).sum() / relative_result.shape[0])
        result["relative < 0.001"] = round_format.format(
            100 * (relative_result < 0.001).sum() / relative_result.shape[0])
        result["relative < 0.0001"] = round_format.format(
            100 * (relative_result < 0.0001).sum() / relative_result.shape[0])
    return result


def torch_load(data, name="data"):
    data = standardize_path(data)
    check_file_safety(data)
    if data.endswith(".bin"):
        data = TensorBinFile(data).get_data().float()
    else:
        data = torch.load(data, weights_only=True).float()
    data_flatten = data.flatten()
    info_msg = f"Loaded: {name}.shape = {list(data.shape)}, After flatten: {name}.shape = {list(data_flatten.shape)}"
    logging.info(info_msg)
    return data, data_flatten


def torch_view(data1, data2):
    logging.info("==================================== data1 ====================================")
    logging.info(data1.shape)
    logging.info(data1)
    logging.info("==================================== data2 ====================================")
    logging.info(data2.shape)
    logging.info(data2)
    logging.info("===============================================================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str, help='path to the first tensor file')
    parser.add_argument('file2', type=str, help='path to the second tensor file')
    parser.add_argument("-p", "--print", default=False, help="print the tensor or not", action="store_true")

    args = parser.parse_known_args()[0]
    if np.sum([args.file1 is None, args.file2 is None]) < 2:
        logging.error("at least 2 data are required")
        
    tensor1, tensor1_flatten = torch_load(args.file1, name="data1_flatten") if args.file1 else None
    tensor2, tensor2_flatten = torch_load(args.file2, name="data2_flatten") if args.file2 else None
    if tensor1_flatten.shape[0] != tensor2_flatten.shape[0]:
        logging.error("The number of elements in two tensor is not equal!!!")

    all_result = OrderedDict()
    if tensor1_flatten is not None and tensor2_flatten is not None:
        if args.print:
            torch_view(tensor1, tensor2)
        all_result["data1 <-> data2"] = compare(tensor1_flatten, tensor2_flatten)

    index = list(next(iter(all_result.values())).keys())  # Keep printing order

    logging.info(pd.DataFrame(all_result, index=index).to_markdown())
