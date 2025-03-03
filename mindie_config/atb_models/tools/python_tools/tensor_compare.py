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

sys.path.append(os.path.dirname(__file__))
from tensor_file import read_tensor  # NOQA: E402


def main():
    tensor1 = read_tensor(sys.argv[1])
    tensor2 = read_tensor(sys.argv[2])

    logger.info(f"tensor1:{tensor1}")
    logger.info(f"tensor2:{tensor2}")
    logger.info(f"tensor1.shape:{tensor1.shape}, dtype:{tensor1.dtype}")
    logger.info(f"tensor2.shape:{tensor2.shape}, dtype:{tensor2.dtype}")

    tensor1 = tensor1.to(torch.float64)
    tensor2 = tensor2.to(torch.float64)

    sub_tensor = tensor1 - tensor2
    abs_tensor = sub_tensor.abs()

    absolute_err = 0
    avg_cosine_similarity = 0
    max_relative_err = 0
    
    if abs_tensor.numel() != 0:
        absolute_err = abs_tensor.type(torch.float64).sum() / abs_tensor.numel()
        cosine_similarity_tensor = torch.cosine_similarity(tensor1, tensor2, dim=0)
        avg_cosine_similarity = cosine_similarity_tensor.abs().sum() / cosine_similarity_tensor.numel()
        div_tensor = tensor2.abs()
        div_tensor.clamp_(1e-6)
        relative_err_tensor = torch.div(abs_tensor, div_tensor)
        max_relative_err = torch.max(relative_err_tensor)

    if absolute_err.item() > 0.001 or max_relative_err.item() > 0.001 :
        logger.info("NOT EQUAL")
        logger.info("Absolute error: ")
        logger.info(absolute_err)
        logger.info("Average cosine similarity:")
        logger.info(avg_cosine_similarity)
        logger.info("Max relative error: ")
        logger.info(max_relative_err)
    else:
        logger.info("EQUAL")


if __name__ == "__main__":
    main()
