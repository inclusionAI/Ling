# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.

import argparse
import json
import os
import time
import open_clip
import torch
import torch_npu
import torch.nn.functional as F

from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.utils.log import logger
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="device for torch.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14",
        help="open clip model name",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./open_clip_pytorch_model.bin",
        help="open clip model weights",
    )
    parser.add_argument(
        "--image_info",
        type=str,
        default="./image_info.json",
        help="Image_info.json file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./cocoTest/",
        help="dataset path for precision test.",
    )
    return parser.parse_args()


def set_torch_env(device_ids):
    torch_npu.npu.set_device(int(device_ids))
    torch.npu.set_compile_mode(jit_compile=False)


def clip_score(model_clip, tokenizer, preprocess, model_answer, image_file):
    imgs = []
    texts = []

    image = safe_open_image(Image, image_file)
    img = preprocess(image).unsqueeze(0).npu()
    image.close()
    imgs.append(img)
    text = tokenizer([model_answer]).npu()
    texts.append(text)

    img = torch.cat(imgs)  # [bs, 3, 224, 224]
    text = torch.cat(texts)  # [bs, 77]

    with torch.no_grad():
        text_ft = model_clip.encode_text(text).float()
        img_ft = model_clip.encode_image(img).float()
        score = F.cosine_similarity(img_ft, text_ft).squeeze()

    return score.cpu()


def main():
    args = parse_arguments()
    set_torch_env(args.device)

    t_b = time.time()
    logger.info("Load clip model...")
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.model_weights_path, device=f"npu:{args.device}")
    model_clip.eval()
    logger.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")

    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    with safe_open(args.image_info, "r", override_flags=os.O_RDONLY) as f:
        image_info = json.load(f)

    t_b = time.time()

    logger.info("Calc clip score...")
    all_scores = []
    for image_file, model_answer in image_info.items():
        # 单个图片  单个answer
        image_file_path = os.path.join(args.dataset_path, image_file)
        logger.info(f"cur image file: {image_file_path}")
        image_score = clip_score(model_clip, tokenizer, preprocess, model_answer, image_file_path)
        logger.info(f"image_score: {image_score}")
        all_scores.append(image_score)
    all_scores_mean = torch.mean(torch.tensor(all_scores))
    logger.info(f"平均分：{all_scores_mean=}")
    logger.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")


if __name__ == '__main__':
    main()
