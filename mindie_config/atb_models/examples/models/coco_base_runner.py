# Copyright 2024 Huawei Technologies Co., Ltd
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

import argparse
import json
import os
import torch
from tqdm import tqdm
from atb_llm.utils.file_utils import safe_open, check_file_safety
from atb_llm.utils.log import logger

torch.manual_seed(1234)
OUTPUT_JSON_PATH = "./gpu_coco_predict.json"


class CocoBaseRunner:
    def __init__(self, model_path, image_path, **kwargs):
        parser = argparse.ArgumentParser(description="Demo")

        parser.add_argument(
            "--model_path", default=model_path, help="Model and tokenizer path."
        )
        parser.add_argument(
            "--image_path", default=image_path, help="Image path for inference."
        )

        logger.info(f"===== model_path: {model_path}")
        logger.info(f"===== image_path: {image_path}")

        self.args = parser.parse_args()
        self.image_answer = {}
        # set save_output parameters
        self.override_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        self.encoding = "utf-8"
        self.indent = None
        self.ensure_ascii = True

    @staticmethod
    def save_output(
        sorted_dict,
        override_flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        encoding="utf-8",
        indent=None,
        ensure_ascii=True,
    ):
        if not os.path.exists(OUTPUT_JSON_PATH):
            with safe_open(
                OUTPUT_JSON_PATH, "w", override_flags=override_flags, encoding=encoding
            ) as fw:
                json.dump(sorted_dict, fw, ensure_ascii=ensure_ascii, indent=indent)
        else:
            with safe_open(OUTPUT_JSON_PATH, "r") as f:
                old_data = json.load(f)
            old_data.update(sorted_dict)
            sorted_dict = dict(sorted(old_data.items()))
            with safe_open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fw:
                json.dump(sorted_dict, fw, ensure_ascii=ensure_ascii, indent=indent)
        logger.info("run run_coco_gpu.py finish! output file: ./gpu_coco_predict.json")

    # prepare model and data
    def prepare(self):
        raise NotImplementedError()

    def process(self, img_path, img_name):
        raise NotImplementedError()

    def run(self):
        model_path = self.args.model_path
        image_path = self.args.image_path
        check_file_safety(image_path, is_check_file_size=False)
        if os.path.exists(model_path) and os.path.exists(image_path):
            images_list = os.listdir(image_path)

            self.prepare()  # need customized implementation

            for _, img_name in enumerate(tqdm(images_list)):
                img_path = os.path.join(image_path, img_name)

                self.process(img_path, img_name)  # need customized implementation

            sorted_dict = dict(sorted(self.image_answer.items()))
            torch.cuda.empty_cache()
            self.save_output(
                sorted_dict,
                override_flags=self.override_flags,
                encoding=self.encoding,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
            )

        else:
            logger.info("model_path or image_path not exist")
