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
#include <vector>
#include <gtest/gtest.h>
#include "atb_speed/log.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "secodeFuzz.h"
#include "../utils/fuzz_utils.h"

namespace atb_speed {
TEST(CommonPositionalEmbeddingFusionDTFuzz, RotaryPositionEmbedding)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "CommonPositionalEmbeddingFusionDTFuzz";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int RotaryTypes[] = {0, 1, 2};
        int HeadNums[] = {1, 16, 24, 48};
        int HeadDims[] = {128, 256};
        int KvHeadNums[] = {1, 16, 24, 48};
        atb_speed::common::RotaryPositionEmbeddingParam param;
        param.rotaryType = static_cast<atb_speed::common::RotaryType>(
            *(unsigned int *) DT_SetGetNumberEnum(
                &g_Element[fuzzIndex++], 0, RotaryTypes, 3));
        param.isFA = FuzzUtil::GetRandomBool(fuzzIndex);
        param.headNum = *(int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 1, HeadNums, 4);
        param.headDim = *(int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 1, HeadDims, 2);
        param.kvHeadNum = *(int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 1, KvHeadNums, 4);

        atb::Node PositionalEmbeddingNode;
        RotaryPositionEmbedding(param, &PositionalEmbeddingNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}