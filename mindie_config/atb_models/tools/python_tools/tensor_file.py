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
import numpy
import torch
from atb_llm.utils import file_utils

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []

        self.__parse_bin_file()

    def get_tensor(self):
        if self.dtype == 0:
            dtype = numpy.float32
        elif self.dtype == 1:
            dtype = numpy.float16
        elif self.dtype == 2:  # int8
            dtype = numpy.int8
        elif self.dtype == 3:  # int32
            dtype = numpy.int32
        elif self.dtype == 9:  # int64
            dtype = numpy.int64
        elif self.dtype == 12:
            dtype = numpy.bool8
        elif self.dtype == 27:
            tensor = torch.frombuffer(self.obj_buffer, dtype=torch.bfloat16)
            tensor = tensor.view(self.dims)
            return tensor
        else:
            raise NotImplementedError(f"unsupport dtype: {self.dtype}")
        tensor = torch.tensor(numpy.frombuffer(self.obj_buffer, dtype=dtype))
        tensor = tensor.view(self.dims)
        return tensor

    def __parse_bin_file(self):
        with file_utils.safe_open(self.file_path, "rb") as fd:
            file_data = fd.read()

            begin_offset = 0
            for i, val in enumerate(file_data):
                if val == ord("\n"):
                    line = file_data[begin_offset: i].decode("utf-8")
                    begin_offset = i + 1
                    fields = line.split("=")
                    attr_name = fields[0]
                    attr_value = fields[1]
                    if attr_name == ATTR_END:
                        self.obj_buffer = file_data[i + 1:]
                        break
                    elif attr_name.startswith("$"):
                        self.__parse_system_atrr(attr_name, attr_value)
                    else:
                        self.__parse_user_attr(attr_name, attr_value)
                        pass

    def __parse_system_atrr(self, attr_name, attr_value):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.obj_len = int(attr_value)
        elif attr_name == ATTR_OBJECT_PREFIX:

            pass

    def __parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            self.dims = [int(i) for i in self.dims]


def read_tensor(file_path):
    file_path = file_utils.standardize_path(file_path)
    file_utils.check_file_safety(file_path, 'r', is_check_file_size=False)
    if file_path.endswith(".bin"):
        tensor_bin = TensorBinFile(file_path)
        return tensor_bin.get_tensor()
    else:
        try:
            tensor = list(torch.load(file_path, weights_only=True).state_dict().values())[0]
        except (IndexError, AttributeError):
            tensor = torch.load(file_path, weights_only=True)
        return tensor
