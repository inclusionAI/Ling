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
#ifndef ATB_TORCH_BASEOPERATION_H
#define ATB_TORCH_BASEOPERATION_H
#include <string>
#include <unordered_map>
#include <torch/script.h>
#include <atb/atb_infer.h>
#include "operation.h"
#include "atb_speed/log.h"

namespace atb_torch {
class BaseOperation : public Operation {
public:
    BaseOperation(const std::string &opType, const std::string &opParam, const std::string &opName);
    ~BaseOperation() override;
    std::string GetOpType() const;
    std::string GetOpParam() const;

protected:
    std::string opType_;
    std::string opParam_;
};
} // namespace atb_torch
#endif