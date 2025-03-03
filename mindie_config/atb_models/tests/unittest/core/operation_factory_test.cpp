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
#include "atb/operation.h"
#include "atb_speed/utils/operation_factory.h"
#include "layers/post_process.h"
#include "nlohmann/json.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace atb_speed;

TEST(OperationFactory, RegisterShouldReturnTrueWhenGivenUniqueOperationName)
{
    bool firstTimeRegister = OperationFactory::Register("SampleLayerCreate_1", &atb_speed::common::SampleLayerCreate);
    ASSERT_EQ(firstTimeRegister, true);

    bool DuplicateRegister = OperationFactory::Register("SampleLayerCreate_1", &atb_speed::common::SampleLayerCreate);
    ASSERT_EQ(DuplicateRegister, false);
}

TEST(OperationFactory, CreateOperationByFunctionWouldNotGetNullptrWhenGivenCorrectParam)
{
    std::string param = R"(
    {
    "temperature": 0.95,
    "topK": 50,
    "randSeed": 12345
    }
    )";
    nlohmann::json paramJson = nlohmann::json::parse(param);
    atb::Operation *operation_ = atb_speed::common::SampleLayerCreate(paramJson);
    ASSERT_NE(operation_, nullptr);
}

TEST(OperationFactory, CreateOperationByCreateOperationWouldNotGetNullptrWhenGivenCorrectParam)
{
    std::string param = R"(
    {
    "temperature": 0.95,
    "topK": 50,
    "randSeed": 12345
    }
    )";
    nlohmann::json paramJson = nlohmann::json::parse(param);
    atb::Operation *operation_ = OperationFactory::CreateOperation("common_SampleLayerCreate", paramJson);
    ASSERT_NE(operation_, nullptr);
}