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
#ifndef ATB_TORCH_OPERATION_FACTORY_H
#define ATB_TORCH_OPERATION_FACTORY_H
#include <map>
#include <string>
#include <functional>
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_torch {
using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &opParamJson)>;

class OperationFactory {
public:
    static OperationFactory &Instance();
    void RegisterOperation(const std::string &opType, OperationCreateFunc func);
    atb::Operation *CreateOperation(const std::string &opType, const std::string &opParam);
    std::vector<std::string> SupportOperations() const;

private:
    std::map<std::string, OperationCreateFunc> operationCreateFuncMap_;
};

#define REGISTER_OPERATION(opType, operationCreateFunc)     \
    struct Register##_##opType##_##operationCreateFunc {    \
        inline Register##_##opType##_##operationCreateFunc() noexcept \
        {  \
            atb_torch::OperationFactory::Instance().RegisterOperation(#opType, operationCreateFunc); \
        }  \
    } static g_register_##opType##operationCreateFunc
} // namespace atb_torch
#endif