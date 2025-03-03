#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import argparse
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from atb_speed.common.performance.base import PerformanceTest
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


class LMLauncher(Launcher):
    """
    LMLauncher
    """

    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = safe_get_tokenizer_from_pretrained(
                                            self.model_path, trust_remote_code=self.trust_remote_code, use_fast=False)
        model = safe_get_model_from_pretrained(self.model_path, 
                                            trust_remote_code=self.trust_remote_code).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    args = get_args()
    trust_remote_code_status = args.trust_remote_code
    atb_speed_config.init_config("template.ini")
    performance_test = PerformanceTest(LMLauncher(device_ids="0", trust_remote_code=trust_remote_code_status,))
    performance_test.warm_up()
    performance_test.run_test()