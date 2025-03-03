# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from PIL import Image
from transformers import CLIPImageProcessor
from atb_llm.utils.multimodal_utils import safe_open_image
from ..base.model_utils import safe_from_pretrained


def expand2square(input_image, background_color):
    width, height = input_image.size
    if height > width:
        result = Image.new(input_image.mode, (height, height), background_color)
        result.paste(input_image, ((height - width) // 2, 0))
        return result
    elif height < width:
        result = Image.new(input_image.mode, (width, width), background_color)
        result.paste(input_image, (0, (width - height) // 2))
        return result
    else:
        return input_image


class DataProcessorYiVl:
    def __init__(self, vision_path, trust_remote_code, **kwargs):
        self.image_processor = safe_from_pretrained(CLIPImageProcessor, vision_path, 
                                        trust_remote_code=trust_remote_code)

    def preprocess_image(self, config, image_path):
        image = safe_open_image(Image, image_path).convert('RGB')
        if getattr(config, "image_aspect_ratio", None) == "pad":
            backgrorund = tuple(int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, backgrorund)
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        image.close()
        return pixel_values


