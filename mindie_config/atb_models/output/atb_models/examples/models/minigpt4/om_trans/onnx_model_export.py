# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import argparse

from PIL import Image
import torch

from atb_llm.utils.log import logger
from atb_llm.utils.file_utils import check_file_safety

from minigpt4.common.registry import registry
from minigpt4.models.eva_vit_model import MiniGPT4ImageEmbedding


def parse_args():
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--onnx-model-dir",
        type=str,
        default="./transfer_model_onnx",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="../test_image/01.jpg",
        help="Location image path",
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    image_path = args.image_path
    onnx_model_dir = args.onnx_model_dir

    if not os.path.exists(onnx_model_dir):
        os.makedirs(onnx_model_dir)

    check_file_safety(image_path, is_check_file_size=False)
    check_file_safety(onnx_model_dir, is_check_file_size=False)
    onnx_model_path = os.path.join(onnx_model_dir, "eva_vit_g.onnx")
    logger.info(f'onnx_model_path:{onnx_model_path}')

    model = MiniGPT4ImageEmbedding()

    vis_processor = registry.get_processor_class("blip2_image_eval").from_config()

    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0)
    logger.info(f'input size:{image.size()}')  # input size: torch.Size([1, 224, 224, 3])

    # onnx model export
    torch.onnx.export(
        model,  # model being run
        image,  # model input (or a tuple for multiple inputs)
        onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={'input': {0: 'batch'}},
    )

    logger.info("====== export onnx model successfully! ======")

