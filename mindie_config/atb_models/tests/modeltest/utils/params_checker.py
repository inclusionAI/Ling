#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
"""
params_checker
"""

import json
import sys

from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log.logging import logger


def check():
    ranktable_path = sys.argv[1]
    param_world_size = sys.argv[2]
    param_nnodes = sys.argv[3]
    param_master_addr = sys.argv[4]
    with safe_open(ranktable_path, "r") as f:
        ranktable = json.load(f)
    world_size = 0
    server_list = ranktable["server_list"]
    for server in server_list:
        world_size += len(server["device"])
    nnodes = ranktable["server_count"]
    master_addr = ""

    for server in server_list:
        for device in server["device"]:
            if device["rank_id"] == "0":
                master_addr = server["server_id"]

    if str(world_size) != param_world_size:
        logger.error("World size does not match with ranktable file")

    if nnodes != param_nnodes:
        logger.error("Node num does not match with ranktable file")
    
    if master_addr != param_master_addr:
        logger.error("Master address does not match with ranktable file")


if __name__ == "__main__":
    check()