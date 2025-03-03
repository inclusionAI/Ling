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
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "chatglm2/6b/model/flash_attention_model.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace atb_speed;

TEST(ModelFactory, RegisterShouldReturnTrueWhenGivenUniqueModelName)
{
    bool firstTimeRegister = ModelFactory::Register("ChatGlm2CommonModelFa_1", [](const std::string &param) {
        return std::make_shared<atb_speed::chatglm2_6b::ChatGlm2CommonModelFa>(param);
    });
    ASSERT_EQ(firstTimeRegister, true);

    bool duplicateRegister = ModelFactory::Register("ChatGlm2CommonModelFa_1", [](const std::string &param) {
        return std::make_shared<atb_speed::chatglm2_6b::ChatGlm2CommonModelFa>(param);
    });
    ASSERT_EQ(duplicateRegister, false);
}

TEST(ModelFactory, CreateModelByClassConstructorWouldNotGetNullptrWhenGivenCorrectParam)
{
    std::string param = R"(
    {
    "rmsNormEps": 0.0,
    "numHeadsPerPartition": 0,
    "hiddenSizePerHead": 0,
    "numGroupsPerPartition": 0,
    "transKey": true,
    "quantmodel": false,
    "isSparse": false,
    "layerNum": 0,
    "correctNodeId": -1,
    "residualAddScale": 0.0,
    "qkvInputScale": [],
    "qkvInputOffset": [],
    "denseInputScale": [],
    "denseInputOffset": [],
    "selfLnInputScale": [],
    "selfLnInputOffset": [],
    "ffnOutInputScale": [],
    "ffnOutInputOffset": [],
    "preScale": [],
    "postScale": [],
    "rank": 0,
    "rankSize": 1,
    "backend": "hccl",
    "isEncoder": false,
    "offsetX": [],
    "compressInfo": []
    }
    )";
    std::shared_ptr<atb_speed::Model> model_ = std::make_shared<atb_speed::chatglm2_6b::ChatGlm2CommonModelFa>(param);
    ASSERT_NE(model_, nullptr);
}

TEST(ModelFactory, CreateModelByCreateInstanceWouldNotGetNullptrWhenGivenCorrectParam)
{
    std::string param = R"(
    {
    "rmsNormEps": 0.0,
    "numHeadsPerPartition": 0,
    "hiddenSizePerHead": 0,
    "numGroupsPerPartition": 0,
    "transKey": true,
    "quantmodel": false,
    "isSparse": false,
    "layerNum": 0,
    "correctNodeId": -1,
    "residualAddScale": 0.0,
    "qkvInputScale": [],
    "qkvInputOffset": [],
    "denseInputScale": [],
    "denseInputOffset": [],
    "selfLnInputScale": [],
    "selfLnInputOffset": [],
    "ffnOutInputScale": [],
    "ffnOutInputOffset": [],
    "preScale": [],
    "postScale": [],
    "rank": 0,
    "rankSize": 1,
    "backend": "hccl",
    "isEncoder": false,
    "offsetX": [],
    "compressInfo": []
    }
    )";
    std::shared_ptr<atb_speed::Model> model_ = ModelFactory::CreateInstance("ChatGlm2CommonModelFa_1", param);
    ASSERT_NE(model_, nullptr);
}