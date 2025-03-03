#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
"""
gpt-neox-20b
@create: 2024/1/24 19:32
@since: 2024/1/24 19:32
"""
import argparse
import os

from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import ParallelLauncher, Launcher
from atb_speed.common.performance.base import PerformanceTest
from atb_speed.common.precision import get_precision_test_cls
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting LLM on Ascend")
    parser.add_argument(
        "--task",
        type=str,
        default='inference',
        choices=['inference', 'precision', 'performance'],
        help="Specify the task in which to run the script"
    )
    args = parser.parse_args()
    return args


class LMParallel(ParallelLauncher):
    """
    多卡推理launcher
    """

    def init_model(self):

        tokenizer = safe_get_tokenizer_from_pretrained(self.model_path, padding_side='left')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        part_model_path = os.path.join(self.model_path, 'part_model', str(self.local_rank))
        model = safe_get_model_from_pretrained(part_model_path, trust_remote_code=False)
        model = model.half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


class LM(Launcher):
    """
    单卡推理launcher
    """

    def init_model(self):
        """
        模型初始化
        :return:
        """

        tokenizer = safe_get_tokenizer_from_pretrained(self.model_path, padding_side='left')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = safe_get_model_from_pretrained(self.model_path, trust_remote_code=False).half().to(self._device)
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


def demo_perf(launcher: Launcher):
    """
    :param launcher:
    :return:
    """
    performance_test = PerformanceTest(launcher)
    performance_test.warm_up()
    performance_test.run_test()


def demo_inference(launcher: Launcher):
    """
    :param launcher:
    :return:
    """
    param_dict = {"max_new_tokens": 64, "do_sample": False, "repetition_penalty": 1.1}
    launcher.logger.info("---------------warm-up---------------")
    launcher.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->', param_dict)

    launcher.logger.info("---------------inference---------------")
    launcher.infer('How to learn a new language?', param_dict)

    launcher.logger.info("---------------batch---------------")
    query_list = [
        "How to learn a new language?",
        'The CEO of Google is',
    ]
    launcher.infer_batch(query_list, param_dict)


TASK_MAP = {
    "inference": demo_inference,
    "precision": demo_ceval,
    "performance": demo_perf
}


def main():
    args = parse_args()
    atb_speed_config.init_config("config.ini")
    if atb_speed_config.model.device_num > 1:
        launcher = LMParallel()
    else:
        launcher = LM()
    TASK_MAP.get(args.task)(launcher)


if __name__ == "__main__":
    main()
