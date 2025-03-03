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
import sys
import os
import torch
from atb_llm.utils.log import logger
from atb_llm.utils import file_utils

sys.path.append(os.path.dirname(__file__))
from tensor_file import read_tensor  # NOQA: E402


def main():
    tensor = read_tensor(sys.argv[1])
    logger.info(f"tensor:{tensor}")
    tensor_save_path = file_utils.standardize_path(sys.argv[2])
    file_utils.check_file_safety(tensor_save_path, 'w', is_check_file_size=False)
    torch.save(tensor, tensor_save_path)


if __name__ == "__main__":
    main()
