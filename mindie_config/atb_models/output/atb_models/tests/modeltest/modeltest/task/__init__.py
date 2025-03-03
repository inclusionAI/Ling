# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import importlib
from modeltest.api.task import Task


def get_precision_task_cls(task_name):
    task_cls_name = task_name.split("_")[0]
    module = importlib.import_module(f".{task_cls_name}", package=__name__)
    task_cls_map = {
        "boolq": "BoolQPrecisionTask",
        "ceval_few_shots": "CEvalFewShotsPrecisionTask",
        "ceval_zero_shot": "CEvalZeroShotPrecisionTask",
        "cmmlu": "CMMLUPrecisionTask",
        "gsm8k": "GSM8KPrecisionTask",
        "humaneval": "HumanEvalPrecisionTask",
        "humanevalx": "HumanEvalXPrecisionTask",
        "longbench": "LongBenchPrecisionTask",
        "longbench_e": "LongBenchPrecisionTask",
        "mmlu_few_shots": "MMLUFewShotsPrecisionTask",
        "mmlu_zero_shot": "MMLUZeroShotPrecisionTask",
        "needlebench": "NeedleBenchPrecisionTask",
        "textvqa": "TextVQAPrecisionTask",
        "truthfulqa": "TruthfulQAPrecisionTask",
        "videobench": "VideoBenchPrecisionTask",
        "vocalsound": "VocalSoundPrecisionTask"

    }
    return getattr(module, task_cls_map.get(task_name, ""))


def get_task_cls(task_config_path):
    task_config = Task.parse_config(task_config_path)
    func_name = f"get_{task_config.task_type}_task_cls"
    try:
        func = globals()[func_name]
        if callable(func):
            return func(task_config.task_name)(task_config)
        else:
            raise TypeError(f"'{task_config.task_type}' is not supported.")
    except KeyError as e:
        raise AttributeError(f"No task function named '{func_name}' found.") from e
