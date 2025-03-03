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
#include "atb_speed/utils/operation_factory.h"
#include "atb_speed/log.h"

namespace atb_speed {
bool OperationFactory::Register(const std::string &operationName, CreateOperationFuncPtr createOperation)
{
    auto it = OperationFactory::GetRegistryMap().find(operationName);
    if (it != OperationFactory::GetRegistryMap().end()) {
        ATB_SPEED_LOG_WARN(operationName << " operation already exists, but the duplication doesn't matter.");
        return false;
    }
    OperationFactory::GetRegistryMap()[operationName] = createOperation;
    return true;
}

atb::Operation *OperationFactory::CreateOperation(const std::string &operationName, const nlohmann::json &param)
{
    auto it = OperationFactory::GetRegistryMap().find(operationName);
    if (it != OperationFactory::GetRegistryMap().end()) {
        if (it->second == nullptr) {
            ATB_SPEED_LOG_ERROR("find operation error: " << operationName);
            return nullptr;
        }
        ATB_SPEED_LOG_DEBUG("find operation: " << operationName);
        return it->second(param);
    }
    ATB_SPEED_LOG_WARN("OperationName: " << operationName << " not find in operation factory map");
    return nullptr;
}

std::unordered_map<std::string, CreateOperationFuncPtr> &OperationFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateOperationFuncPtr> operationRegistryMap;
    return operationRegistryMap;
}
} // namespace atb_speed
