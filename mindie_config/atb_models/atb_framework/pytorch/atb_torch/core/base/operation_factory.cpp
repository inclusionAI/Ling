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
#include "operation_factory.h"

namespace atb_torch {
OperationFactory &OperationFactory::Instance()
{
    static OperationFactory instance;
    return instance;
}

void OperationFactory::RegisterOperation(const std::string &opType, OperationCreateFunc func)
{
    operationCreateFuncMap_[opType] = func;
}

atb::Operation *OperationFactory::CreateOperation(const std::string &opType, const std::string &opParam)
{
    auto it = operationCreateFuncMap_.find(opType);
    if (it == operationCreateFuncMap_.end()) {
        ATB_SPEED_LOG_ERROR("create atb operation fail, not find opType:" << opType);
        return nullptr;
    }

    size_t maxParamLength = 200000;
    if (opParam.size() > maxParamLength) {
        ATB_SPEED_LOG_ERROR("create atb operation fail, op_param's length (" << opParam.size()
            << ") should be smaller than " << maxParamLength);
        return nullptr;
    }

    nlohmann::json opParamJson;
    try {
        opParamJson = nlohmann::json::parse(opParam);
    } catch (...) {
        return nullptr;
    }

    return it->second(opParamJson);
}

std::vector<std::string> OperationFactory::SupportOperations() const
{
    std::vector<std::string> operationTypes;
    for (auto &it : operationCreateFuncMap_) {
        operationTypes.push_back(it.first);
    }
    return operationTypes;
}
} // namespace atb_torch