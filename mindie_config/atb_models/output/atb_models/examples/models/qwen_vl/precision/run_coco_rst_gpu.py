# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import json
import os
from tqdm import tqdm
from atb_llm.utils import argument_utils
from atb_llm.utils.file_utils import safe_open, standardize_path
from atb_llm.utils.file_utils import check_file_safety, safe_listdir
from atb_llm.models.base.model_utils import (
    safe_get_tokenizer_from_pretrained,
    safe_get_model_from_pretrained,
)


def validate_path_new(value: str):
    check_file_safety(standardize_path(value), mode='a')
    

def parse_args():
    path_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096)
    savepath_validator = argument_utils.StringArgumentValidator()
    savepath_validator.validate_path = validate_path_new
    savepath_validator.create_validation_pipeline()
    parser = argument_utils.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", 
                        required=True, 
                        help="Model and tokenizer path.", 
                        validator=path_validator)
    parser.add_argument("--image_path", 
                        required=True, 
                        help="Image path for inference.",
                        validator=path_validator,
                        )
    parser.add_argument(
        "--results_save_path",
        help="precision test result path",
        default="./gpu_coco_rst.json",
        validator=savepath_validator,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_file_safety(args.image_path, is_check_file_size=False)
    check_file_safety(args.model_path, is_check_file_size=False)
    images_list = safe_listdir(args.image_path)
    tokenizer = safe_get_tokenizer_from_pretrained(
        args.model_path, trust_remote_code=False
    )
    model = safe_get_model_from_pretrained(
        args.model_path, device_map="cuda", trust_remote_code=False, fp16=True
    ).eval()
    gpu_rst = {}
    for _, img_name in enumerate(tqdm(images_list)):
        img_path = os.path.join(args.image_path, img_name)
        query = tokenizer.from_list_format(
            [
                {"image": img_path},
                {"text": "Generate the caption in English with grounding:"},
            ]
        )
        inputs = tokenizer(query, return_tensors="pt")
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        gpu_rst[img_name] = response.split("grounding:")[-1]
    sorted_dict = dict(sorted(gpu_rst.items()))
    with safe_open(
        args.results_save_path,
        "w",
        override_flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    ) as f:
        json.dump(sorted_dict, f)


if __name__ == "__main__":
    main()
