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
#include <gtest/gtest.h>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(WordEmbeddingDTFuzz, WordEmbedding)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "WordEmbeddingDTFuzzWordEmbedding";

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        std::vector<std::string> backendEnumTable = {"lccl", "hccl"};

        atb_speed::common::WordEmbeddingParam param;
        param.unpadInputs = FuzzUtil::GetRandomBool(fuzzIndex);
        param.tensorParallelInfo.rank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        param.tensorParallelInfo.worldSize = worldSize;
        param.tensorParallelInfo.backend = backendEnumTable[*(int *) DT_SetGetNumberRange( \
            &g_Element[fuzzIndex++], 0, 0, 1)];

        atb::Node wordEmbeddingNode;
        WordEmbedding(param, &wordEmbeddingNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}