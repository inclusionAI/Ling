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
#include <string>
#include <gtest/gtest.h>
#include "atb_speed/utils/match.h"
#include "atb_speed/log.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(MatchDTFuzz, StartsWith)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "MatchDTFuzzStartsWith";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        char textInitValue[] = "zhangpeng";
        char prefixInitValue[] = "zhan";
        // 10 is min, 10000 is max size of text
        const std::string &text = DT_SetGetString(&g_Element[fuzzIndex++], 10, 10000, textInitValue);
        // 5 is min, 10000 is max size of prefix
        const std::string &prefix = DT_SetGetString(&g_Element[fuzzIndex++], 5, 10000, prefixInitValue);
        StartsWith(text, prefix);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(MatchDTFuzz, EndsWith)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "MatchDTFuzzEndsWith";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        char textInitValue[] = "zhangpeng";
        char suffixInitValue[] = "peng";
        // 10 is min, 10000 is max size of text
        const std::string &text = DT_SetGetString(&g_Element[fuzzIndex++], 10, 10000, textInitValue);
        // 5 is min, 10000 is max size of prefix
        const std::string &prefix = DT_SetGetString(&g_Element[fuzzIndex++], 5, 10000, suffixInitValue);
        EndsWith(text, prefix);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}