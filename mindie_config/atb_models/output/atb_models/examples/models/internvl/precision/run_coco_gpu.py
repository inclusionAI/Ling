# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import json
import os
import torch
from tqdm import tqdm
from atb_llm.utils import argument_utils
from atb_llm.utils.argument_utils import BooleanArgumentValidator, ArgumentAction
from atb_llm.utils.file_utils import safe_open, safe_listdir, standardize_path, check_file_safety, MAX_IMAGE_SIZE
from atb_llm.utils.log import logger
from atb_llm.models.internvl.data_preprocess_internvl import load_image
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained, safe_get_model_from_pretrained

torch.manual_seed(1234)
OUTPUT_JSON_PATH = "./gpu_coco_predict.json"


def parse_args():
    path_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096)
    bool_validator = BooleanArgumentValidator()
    parser = argument_utils.ArgumentParser(description="Demo")
    parser.add_argument("--model_path",
                        required=True,
                        help="Model and tokenizer path.",
                        validator=path_validator)
    parser.add_argument("--image_path",
                        required=True,
                        help="Image path for inference.",
                        validator=path_validator)
    parser.add_argument('--trust_remote_code', action=ArgumentAction.STORE_TRUE.value, 
                        validator=bool_validator)
    return parser.parse_args()


def main():
    generation_config = dict(
        num_beams=1,
        max_new_tokens=256,
        do_sample=False,
    )
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path
    trust_remote_code = args.trust_remote_code
    model_name = os.path.split(model_path)[-1]
    image_name = os.path.split(image_path)[-1]
    logger.info(f"===== model_path: {model_name}")
    logger.info(f"===== image_path: {image_name}")
    if os.path.exists(model_path) and os.path.exists(image_path):
        image_path = standardize_path(image_path)
        check_file_safety(image_path, max_file_size=MAX_IMAGE_SIZE)
        images_list = safe_listdir(image_path)
        tokenizer = safe_get_tokenizer_from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model = safe_get_model_from_pretrained(model_path,
                                                device_map="auto",
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=trust_remote_code,
                                                torch_dtype=torch.float16).eval()
        image_answer = {}
        for _, img_name in enumerate(tqdm(images_list)):
            img_path = os.path.join(image_path, img_name)
            pixel_values = load_image(img_path, max_num=6).to(torch.float16).cuda()
            question = '用500字详细描述图片'
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            image_answer[img_name] = response

        sorted_dict = dict(sorted(image_answer.items()))
        torch.cuda.empty_cache()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if not os.path.exists(OUTPUT_JSON_PATH):
            with safe_open(OUTPUT_JSON_PATH, "w", override_flags=flags) as fw:
                json.dump(sorted_dict, fw)
        else:
            with safe_open(OUTPUT_JSON_PATH, "r") as f:
                old_data = json.load(f)
            old_data.update(sorted_dict)
            sorted_dict = dict(sorted(old_data.items()))
            with safe_open(OUTPUT_JSON_PATH, "w", override_flags=flags) as fw:
                json.dump(sorted_dict, fw)
        logger.info("run run_coco_gpu.py finish! output file: ./gpu_coco_predict.json")
    else:
        logger.info("model_path or image_path not exist")


if __name__ == "__main__":
    main()
