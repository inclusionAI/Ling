/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_TORCH_CONFIG_H
#define ATB_TORCH_CONFIG_H
#include <string>
#include <set>
#include <torch/extension.h>

namespace atb_torch {
class Config {
public:
    static Config &Instance();
    bool IsUseTilingCopyStream() const;
    bool IsTorchTensorFormatCast() const;
    bool IsTaskQueueEnable() const;
    uint64_t GetGlobalWorkspaceSize() const;
    void SetGlobalWorkspaceSize(uint64_t size);
    torch::Tensor &GetGlobalWorkspaceTensor();

private:
    Config();
    ~Config();
    bool IsEnable(const char *env, bool enable = false) const;

private:
    bool isUseTilingCopyStream_ = false;
    bool isTorchTensorFormatCast_ = false;
    bool isTaskQueueEnable_ = false;
    uint64_t globalWorkspaceSize_ = 0;
    torch::Tensor globalWorkspaceTensor_;
};
} // namespace atb_torch
#endif