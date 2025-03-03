# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
import open_clip
from PIL import Image
import torch
import torch_npu
import torch.nn.functional as F

from atb_llm.utils import argument_utils
from atb_llm.models.base.model_utils import safe_open_clip_from_pretrained
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.multimodal_utils import safe_load_multimodal_source
from atb_llm.utils.file_utils import check_file_safety
from atb_llm.utils.log import logger
from examples.multimodal_runner import path_validator


def parse_arguments():
    parser = argument_utils.ArgumentParser(description="")
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=100)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="device for torch.",
        validator=string_validator,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14",
        help="open clip model name",
        validator=string_validator,
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./open_clip_pytorch_model.bin",
        help="open clip model weights",
        validator=path_validator,
    )
    parser.add_argument(
        "--image_info",
        type=str,
        default="./image_info.json",
        help="Image_info.json file.",
        validator=path_validator,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./cocoTest/",
        help="dataset path for precision test.",
        validator=path_validator,
    )
    return parser.parse_args()


def set_torch_env(device_ids):
    torch_npu.npu.set_device(int(device_ids))
    torch.npu.set_compile_mode(jit_compile=False)


def clip_score(model_clip, tokenizer, preprocess, model_answer, image_file):
    imgs = []
    texts = []

    image_loaded = safe_load_multimodal_source(Image.open, image_file)

    img = preprocess(image_loaded).unsqueeze(0).npu()
    imgs.append(img)
    text = tokenizer([model_answer]).npu()
    texts.append(text)

    img = torch.cat(imgs)  # [bs, 3, 224, 224]
    text = torch.cat(texts)  # [bs, 77]

    with torch.no_grad():
        text_ft = model_clip.encode_text(text).float()
        img_ft = model_clip.encode_image(img).float()
        score = F.cosine_similarity(img_ft, text_ft).squeeze()

    image_loaded.close()

    return score.cpu()


def main():
    args = parse_arguments()
    set_torch_env(args.device)

    t_b = time.time()
    logger.info("Load clip model...")
    check_file_safety(args.model_weights_path, is_check_file_size=False)
    model_clip, _, preprocess = safe_open_clip_from_pretrained(open_clip.create_model_and_transforms,
                                                               args.model_name, pretrained=args.model_weights_path,
                                                               device=f"npu:{args.device}")
    model_clip.eval()
    logger.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")

    tokenizer = safe_open_clip_from_pretrained(open_clip.get_tokenizer, "ViT-H-14")
    with safe_open(args.image_info, "r", override_flags=os.O_RDONLY) as f:
        image_info = json.load(f)

    t_b = time.time()

    logger.info("Calc clip score...")
    all_scores = []
    check_file_safety(args.dataset_path, is_check_file_size=False)
    for image_file, model_answer in image_info.items():
        # 单个图片  单个answer
        image_file_path = os.path.join(args.dataset_path, image_file)
        logger.info(f"cur image file: {image_file}")
        image_score = clip_score(model_clip, tokenizer, preprocess, model_answer, image_file_path)
        logger.info(f"{image_score=}")
        all_scores.append(image_score)
    all_scores_mean = torch.mean(torch.tensor(all_scores))
    logger.info(f"平均分：{all_scores_mean=}")
    logger.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")


if __name__ == '__main__':
    main()
