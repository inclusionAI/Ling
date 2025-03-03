/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "atb_speed/utils/statistic.h"

namespace atb_speed {
thread_local Statistic g_statistic;

std::string Statistic::ToString() const
{
    return "totalTime:" + std::to_string(totalTime) + ", createTensorTime:" + std::to_string(createTensorTime) +
           ", planSetupTime:" + std::to_string(planSetupTime) + ", planAsyncTime:" + std::to_string(planAsyncTime) +
           ", planExecuteTime:" + std::to_string(planExecuteTime) +
           ", streamSyncTime:" + std::to_string(streamSyncTime) +
           ", tillingCopyTime:" + std::to_string(tillingCopyTime) +
           ", getBestKernelTime:" + std::to_string(getBestKernelTime) +
           ", kernelExecuteTime:" + std::to_string(kernelExecuteTime) +
           ", kernelCacheHitCount:" + std::to_string(kernelCacheHitCount) +
           ", kernelCacheMissCount:" + std::to_string(kernelCacheMissCount) +
           ", mallocTorchTensorSize:" + std::to_string(mallocTorchTensorSize);
}

void Statistic::Reset()
{
    totalTime = 0;
    createTensorTime = 0;
    planSetupTime = 0;
    planAsyncTime = 0;
    planExecuteTime = 0;
    streamSyncTime = 0;
    tillingCopyTime = 0;
    getBestKernelTime = 0;
    kernelExecuteTime = 0;
    kernelCacheHitCount = 0;
    kernelCacheMissCount = 0;
    mallocTorchTensorSize = 0;
}

Statistic &GetStatistic() { return g_statistic; }
} // namespace atb_speed