# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
import argparse
import json
import os
import torch
import torch_npu
from tqdm import tqdm

from atb_llm.models.base.model_utils import safe_get_model_from_pretrained, safe_get_tokenizer_from_pretrained
from atb_llm.utils.file_utils import safe_open, safe_listdir
from atb_llm.utils.log import logger

torch.manual_seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path",
                        required=True,
                        type=str,
                        help="Model and tokenizer path.")
    parser.add_argument("--image_path",
                        type=str,
                        required=True,
                        help="Image path for inference.")
    parser.add_argument("--output_file",
                        type=str,
                        default="./torch_npu_coco_predict.json",
                        help="Output file for results.")
    parser.add_argument('--trust_remote_code', action='store_true')
    return parser.parse_args()


def main():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path
    output_file = args.output_file
    logger.info(f"===== model_path: {model_path}")
    logger.info(f"===== image_path: {image_path}")
    logger.info(f"===== output_file: {output_file}")
    if os.path.exists(model_path) and os.path.exists(image_path):
        images_list = safe_listdir(image_path)
        tokenizer = safe_get_tokenizer_from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
        model = safe_get_model_from_pretrained(model_path,
                                        trust_remote_code=args.trust_remote_code).half().npu()
        image_answer = {}
        for _, img_name in enumerate(tqdm(images_list)):
            img_path = os.path.join(image_path, img_name)
            prompt = '<ImageHere>Please describe this image in detail.'
            with torch.no_grad():
                response, _ = model.chat(tokenizer, query=prompt, image=img_path, history=[], do_sample=False)
            image_answer[img_name] = response

        sorted_dict = dict(sorted(image_answer.items()))
        torch.cuda.empty_cache()
        if not os.path.exists(output_file):
            with safe_open(output_file, "w") as fw:
                json.dump(sorted_dict, fw)
        else:
            with safe_open(output_file, "r") as f:
                old_data = json.load(f)
            old_data.update(sorted_dict)
            sorted_dict = dict(sorted(old_data.items()))
            with safe_open(output_file, 'w') as fw:
                json.dump(sorted_dict, fw)
        logger.info(f"run run_coco_gpu.py finish! output file: {output_file}")
    else:
        logger.info("model_path or image_path not exist")


if __name__ == "__main__":
    main()
