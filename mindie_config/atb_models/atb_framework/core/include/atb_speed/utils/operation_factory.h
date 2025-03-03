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
#ifndef ATB_SPEED_UTILS_OPERATION_FACTORY_H
#define ATB_SPEED_UTILS_OPERATION_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "atb/operation.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"


namespace atb_speed {
using CreateOperationFuncPtr = std::function<atb::Operation *(const nlohmann::json &)>;

class OperationFactory {
public:
    static bool Register(const std::string &operationName, CreateOperationFuncPtr createOperation);
    static atb::Operation *CreateOperation(const std::string &operationName, const nlohmann::json &param);
private:
    static std::unordered_map<std::string, CreateOperationFuncPtr> &GetRegistryMap();
};

#define OPERATION_NAMESPACE_STRINGIFY(operationNameSpace) #operationNameSpace
#define REGISTER_OPERATION(nameSpace, operationCreateFunc)                                                   \
        struct Register##_##nameSpace##_##operationCreateFunc {                                              \
            inline Register##_##nameSpace##_##operationCreateFunc()                                          \
            {                                                                                                \
                ATB_SPEED_LOG_DEBUG("register operation " << #nameSpace << "_" << #operationCreateFunc;        \
                OperationFactory::Register(OPERATION_NAMESPACE_STRINGIFY(nameSpace##_##operationCreateFunc), \
                    &(operationCreateFunc)));                                                                 \
            }                                                                                                \
        } static instance_##nameSpace##operationCreateFunc
} // namespace atb_speed
#endif