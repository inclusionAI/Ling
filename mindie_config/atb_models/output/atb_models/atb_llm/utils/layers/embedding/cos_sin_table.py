# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass


@dataclass
class CosSinTable:
    dim = 0
    offset = 0
    rope_given_inv_feq_str = None
    rope_keep_local_base_windows = None
    rope_mscale = 1
    rope_theta = None
    rope_vanilla_theta = None