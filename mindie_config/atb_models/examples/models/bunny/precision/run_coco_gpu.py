# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import json
import os
import torch
from tqdm import tqdm
from PIL import Image

from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.utils.log import logger
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained, safe_get_model_from_pretrained

torch.manual_seed(1234)
OUTPUT_JSON_PATH = "./gpu_coco_predict.json"
torch.set_default_device('cuda')


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path",
                        required=True,
                        help="Model and tokenizer path.")
    parser.add_argument("--image_path",
                        required=True,
                        help="Image path for inference.")
    return parser.parse_args()


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path
    logger.info(f"===== model_path: {model_path}")
    logger.info(f"===== image_path: {image_path}")
    if os.path.exists(model_path) and os.path.exists(image_path):
        images_list = os.listdir(image_path)
        tokenizer = safe_get_tokenizer_from_pretrained(model_path, trust_remote_code=False)
        model = safe_get_model_from_pretrained(model_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                torch_dtype=torch.float16).eval()
        model_type = model.config.model_type
        image_answer = {}
        for _, img_name in enumerate(tqdm(images_list)):
            prompt = '用500字详细描述图片'
            text = f"A chat between a curious user and an artificial intelligence assistant. \
            The assistant gives helpful, detailed, and polite answers to the user's questions. \
            USER: <image>\n{prompt} ASSISTANT:"
            text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            temp_string = text_chunks[1][1:]
            if model_type == "bunny-qwen2":
                temp_string = text_chunks[1]
            input_ids = torch.tensor(text_chunks[0] + [-200] + temp_string, dtype=torch.long).unsqueeze(0).to('cuda')

            # image, sample images can be found in images folder
            img_path = os.path.join(image_path, img_name)
            image = safe_open_image(Image, img_path)
            image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device='cuda')
            image.close()
            # generate
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=256,
                use_cache=True,
                repetition_penalty=1.0  # increase this to avoid chattering
            )[0]

            response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
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
            with safe_open(OUTPUT_JSON_PATH, 'w', override_flags=flags) as fw:
                json.dump(sorted_dict, fw)
        logger.info("run run_coco_gpu.py finish! output file: ./gpu_coco_predict.json")
    else:
        logger.info("model_path or image_path not exist")


if __name__ == "__main__":
    main()