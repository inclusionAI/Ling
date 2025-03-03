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
#include <string>
#include <iostream>
#include <thread>
#include "utils.h"
#include "atb_speed/log.h"
#include "config.h"

namespace atb_torch {
constexpr uint64_t  DEFAULT_WORKSPACE_SIZE = 1024 * 1024 * 600;

Config &Config::Instance()
{
    static Config instance;
    return instance;
}

Config::Config()
{
    isUseTilingCopyStream_ = IsEnable("ATB_USE_TILING_COPY_STREAM");
    isTorchTensorFormatCast_ = IsEnable("ATB_TORCH_TENSOR_FORMAT_CAST");
    const char *taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
    const char *blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
    isTaskQueueEnable_ = !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
                           (blockingEnv != nullptr && std::string(blockingEnv) == "1"));
    SetGlobalWorkspaceSize(DEFAULT_WORKSPACE_SIZE);

    ATB_SPEED_LOG_DEBUG("Config [IsTorchTensorFormatCast:" << isTorchTensorFormatCast_
                        << ", IsUseTilingCopyStream:" << isUseTilingCopyStream_
                        << ", GlobalWorkspaceSize:" << GetGlobalWorkspaceSize()
                        << ", IsTaskQueueEnable:" << isTaskQueueEnable_ << "]");
}

Config::~Config() {}

bool Config::IsEnable(const char *env, bool enable) const
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsUseTilingCopyStream() const { return isUseTilingCopyStream_; }

bool Config::IsTorchTensorFormatCast() const { return isTorchTensorFormatCast_; };

bool Config::IsTaskQueueEnable() const { return isTaskQueueEnable_; }

uint64_t Config::GetGlobalWorkspaceSize() const { return globalWorkspaceSize_; }

void Config::SetGlobalWorkspaceSize(uint64_t size)
{
    if (size == globalWorkspaceSize_) {
        return;
    }

    globalWorkspaceSize_ = size;

    if (size == 0) {
        globalWorkspaceTensor_ = at::Tensor();
        return;
    }

    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;

    constexpr int KB_1 = 1024;
    tensorDesc.shape.dimNum = 2; // 2 dims， KB base
    tensorDesc.shape.dims[0] = KB_1;
    tensorDesc.shape.dims[1] = size / KB_1 + 1;

    globalWorkspaceTensor_ = Utils::CreateAtTensorFromTensorDesc(tensorDesc);
}

torch::Tensor &Config::GetGlobalWorkspaceTensor() { return globalWorkspaceTensor_; }
} // namespace atb_torch