#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import argparse
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from atb_speed.common.precision import get_precision_test_cls
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


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
        tokenizer = safe_get_tokenizer_from_pretrained(self.model_path, 
                                                    trust_remote_code=self.trust_remote_code, use_fast=False)
        model = safe_get_model_from_pretrained(self.model_path, 
                                                    trust_remote_code=self.trust_remote_code).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


def demo_ceval(launcher: Launcher):
    """

    :param launcher:
    :return:
    """
    c_t = get_precision_test_cls()(launcher)
    c_t.run()


if __name__ == '__main__':
    args = get_args()
    trust_remote_code_status = args.trust_remote_code
    atb_speed_config.init_config("template.ini")
    baichuan = BaichuanLM(device_ids="0", trust_remote_code=trust_remote_code_status,)
    demo_ceval(baichuan)