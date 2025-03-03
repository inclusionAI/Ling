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
#include "base_operation.h"
#include "operation_factory.h"

namespace atb_torch {
BaseOperation::BaseOperation(const std::string &opType, const std::string &opParam, const std::string &opName)
    : Operation(opName), opType_(opType), opParam_(opParam)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " construct start, opType:" << opType << ", opParam:" << opParam);
    atbOperation_ = OperationFactory::Instance().CreateOperation(opType, opParam);
    CHECK_THROW(atbOperation_ == nullptr, opName_ << "create atb operation fail, please check opParam");

    ATB_SPEED_LOG_DEBUG(opName_ << " construct end");
}

BaseOperation::~BaseOperation() { ATB_SPEED_LOG_DEBUG(opName_ << " disconstruct"); }

std::string BaseOperation::GetOpType() const { return opType_; }

std::string BaseOperation::GetOpParam() const { return opParam_; }
} // namespace atb_torch
