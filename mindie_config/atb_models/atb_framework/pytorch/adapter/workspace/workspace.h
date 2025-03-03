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
#ifndef ATB_SPEED_UTILS_WORKSPACE_H
#define ATB_SPEED_UTILS_WORKSPACE_H
#include <cstdint>
#include <memory>
#include "buffer_base.h"

namespace atb_speed {

class Workspace {
public:
    Workspace();
    ~Workspace();
    void *GetWorkspaceBuffer(uint64_t bufferSize);

private:
    std::unique_ptr<BufferBase> workspaceBuffer_;
};
} // namespace atb_speed
#endif