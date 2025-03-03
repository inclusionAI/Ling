# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math


def assign(expert_count, world_size):
    per_device = math.ceil(expert_count / world_size)
    assignment = []
    if expert_count % world_size == 0:
        for i in range(world_size):
            assignment.append([i * per_device + j for j in range(per_device)])
    else:
        for i in range(world_size - 1):
            assignment.append([i * per_device + j for j in range(per_device)])
        assignment.append([])
        for i in range(expert_count % world_size):
            assignment[-1].append(per_device * (world_size - 1) + i)
    return assignment