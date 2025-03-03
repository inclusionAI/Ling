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
#include "operations/fusion/linear/linear.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(LinearDTFuzz, FusionLinear)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "LinearDTFuzzFusionLinear";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int LinearQuantTypeEnumTable[] = {0, 1, 2, 3, 4, 5};
        int TransposeTypeEnumTable[] = {-1, 0, 1};
        atb_speed::common::FusionLinearParam param;
        param.quantType = static_cast<atb_speed::common::LinearQuantType>(
            *(unsigned int *) DT_SetGetNumberEnum(
                &g_Element[fuzzIndex++], 0, LinearQuantTypeEnumTable, 6)); // 6 is size of quant type
        param.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);;
        param.hasBias = FuzzUtil::GetRandomBool(fuzzIndex);;
        param.transposeType = static_cast<atb_speed::common::TransposeType>(
            *(int *) DT_SetGetNumberEnum(
                &g_Element[fuzzIndex++], -1, TransposeTypeEnumTable, 3)); // 3 is size of transpose type

        atb::Node linearNode;
        FusionLinear(param, &linearNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}