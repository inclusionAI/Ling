# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from typing import List
import numpy as np
from ..base.input_builder import InputBuilder

_PREFIX_LEN = 7
_TEXT_BOI_POS = 4
_TEXT_EOI_POS = 6


class Glm4vInputBuilder(InputBuilder):
    def __init__(self, tokenizer, config, **kwargs):
        self.config = config
        super().__init__(tokenizer, **kwargs)
    
    def generate_position_ids(self, input_ids):
        if self.config.boi_token_id not in input_ids:
            return range(len(input_ids))
        
        eoi_pos = np.where(np.equal(input_ids, self.config.eoi_token_id))[0][0]
        text_input_ids = input_ids[eoi_pos + 1:]
        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        num_patches = (image_size // patch_size // 2) ** 2
        position_ids = np.arange(len(text_input_ids) + _PREFIX_LEN, dtype=np.int32)
        new_position_ids = []
        new_position_ids.append(np.concatenate(
            (position_ids[:_TEXT_BOI_POS + 1], position_ids[_TEXT_BOI_POS + 1].repeat(num_patches),
             position_ids[_TEXT_EOI_POS:])
        ))
        new_position_ids = np.concatenate(new_position_ids)
        return new_position_ids
