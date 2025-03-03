#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import argparse
from atb_speed.common.launcher import Launcher
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained
from atb_speed.common.config import atb_speed_config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate LLM.')
    parser.add_argument(
        '--trust_remote_code',
        type=bool,
        default=False,
        help='Indicates whether to trust the local executable file.'
    )
    return parser.parse_args()



class BaichuanLM(Launcher):

    def init_model(self):
        """
        模型初始化
        :return:
        """
        pwd = os.path.realpath(os.path.dirname(__file__))
        model_path = os.path.join(pwd, "..", "model")
        tokenizer = safe_get_tokenizer_from_pretrained(model_path, 
                                                    trust_remote_code=self.trust_remote_code, use_fast=False)
        model = safe_get_model_from_pretrained(model_path, 
                                                    trust_remote_code=self.trust_remote_code).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    args = get_args()
    atb_speed_config.init_config("template.ini")
    trust_remote_code_status = args.trust_remote_code
    baichuan = BaichuanLM(device_ids="1", trust_remote_code=trust_remote_code_status,)
    baichuan.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->')

    baichuan.infer('登鹳雀楼->王之涣\n夜雨寄北->')
    baichuan.infer('苹果公司的CEO是')

    query_list = [
        "谷歌公司的CEO是",
        '登鹳雀楼->王之涣\n夜雨寄北->',
        '苹果公司的CEO是',
        '华为公司的CEO是',
        '微软公司的CEO是'
    ]
    baichuan.infer_batch(query_list)