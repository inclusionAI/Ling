# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from ais_bench.infer.interface import InferSession


class ImageEncoderOM:
    def __init__(self, model_path, device):
        self.image_encoder_om = InferSession(device, model_path)  # read local om_file

