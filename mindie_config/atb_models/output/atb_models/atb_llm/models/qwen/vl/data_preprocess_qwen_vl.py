# Copyright (c) Alibaba Cloud.
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the same directory of this file. Changes have
# been made to fit Ascend devices, satisfy Huawei clean code
# regulations and speed up inference.
#
import os
import numpy as np
from PIL import Image

from atb_llm.utils.log import logger
from atb_llm.utils.file_utils import standardize_path
from atb_llm.utils.multimodal_utils import safe_load_multimodal_source


_IMAGE_SIZE = 448


def _supported_image_format(abs_path):
    suffix = os.path.splitext(abs_path)
    return len(suffix) > 0 and suffix[-1] in [".jpg", ".jpeg", ".png"]


def qwen_vl_image_preprocess(image_path):
    SUPPORTED_IMAGE_MODE = "RGB"
    image_path = standardize_path(image_path)

    if _supported_image_format(image_path):
        image = safe_load_multimodal_source(Image.open, image_path)
    else:
        logger.warning(
            "Invalid image path or format (we support png, jpg, jpeg), use white canvas instead."
        )
        image = Image.new(SUPPORTED_IMAGE_MODE, (224, 224), (255, 255, 255))

    if image.mode != SUPPORTED_IMAGE_MODE:
        image = image.convert(SUPPORTED_IMAGE_MODE)

    image = image.resize((_IMAGE_SIZE, _IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    image_npy = np.array(image, dtype=np.float32)
    image.close()

    # the mean and std are from the official implementation of Qwen-VL
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    r, g, b = image_npy[:, :, 0], image_npy[:, :, 1], image_npy[:, :, 2]
    r = (r / 255 - mean[0]) / std[0]
    g = (g / 255 - mean[1]) / std[1]
    b = (b / 255 - mean[2]) / std[2]
    return np.stack([r, g, b])
