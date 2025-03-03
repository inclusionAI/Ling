# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import importlib

metric_map = {
    "acc": "AccMetric",
    "longbench": "LongbenchMetric",
    "pass_k": "PassKMetric", # HumanEval Task
    "truthfulqa": "TruthfulqaMetric"
}


def get_metric_cls(metric_type):
    module = importlib.import_module(f".{metric_type}", package=__name__)
    return getattr(module, metric_map.get(metric_type))