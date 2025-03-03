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


import os
import torch
from PIL import Image
from atb_llm.utils.log import logger
from atb_llm.models.base.model_utils import safe_from_pretrained
from examples.models.coco_base_runner import CocoBaseRunner


class CocoLLaVARunner(CocoBaseRunner):
    def __init__(self, model_path, image_path):
        super().__init__(model_path, image_path)

        llava_type = "llava"
        self.args.llava_type = llava_type
        logger.info(f"===== llava_type: {llava_type}")

        # set save_output parameters
        self.device = 'cuda:0'
        self.override_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        self.encoding = "utf-8"
        self.indent = None
        self.ensure_ascii = True
        self.processor = None
        self.model = None

    def prepare(self):
        model_path = self.args.model_path
        llava_type = self.args.llava_type
        device = self.device

        processor = None
        model = None
        if llava_type == "llava":
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            processor = safe_from_pretrained(LlavaProcessor, model_path, trust_remote_code=False)
            model = safe_from_pretrained(LlavaForConditionalGeneration, model_path,
                                         torch_dtype=torch.float16, device_map=device)
        else:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = safe_from_pretrained(LlavaNextProcessor, model_path, trust_remote_code=False)
            model = safe_from_pretrained(LlavaNextForConditionalGeneration, model_path,
                                         torch_dtype=torch.float16, device_map=device)
        self.processor = processor
        self.model = model


    def process(self, img_path, img_name):

        processor = self.processor
        model = self.model
        image = Image.open(img_path)
        prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
        device = self.device

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        image.close()
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=30)
        response = processor.decode(outputs.cpu()[0], skip_special_tokens=True)

        self.image_answer[img_name] = response.split("ASSISTANT:")[-1]


if __name__ == "__main__":

    llava_model_path = "/data/datasets/llava-1.5-13b-hf" 
    llava_image_path = "/data/datasets/coco_data/val_images" 

    runner = CocoLLaVARunner(llava_model_path, llava_image_path)
    runner.run()