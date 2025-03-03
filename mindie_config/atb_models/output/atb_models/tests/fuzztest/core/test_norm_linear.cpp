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
#include "operations/fusion/norm/norm_linear.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
int g_linearQuantTypeEnumTable[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
int g_normTypeEnumTable[] = {0, 1, 2, 3};
int g_quantTypeEnumTable[] = {0, 1, 2, 3, 4, 5};
int g_transposeTypeEnumTable[] = {-1, 0, 1};
TEST(NormLinearLayerNormDTFuzz, NormLinear)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "NormLinearLayerNormDTFuzzNormLinearLayerNorm";

    DT_FUZZ_START(0, 10000, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        atb_speed::common::NormLinearParam<atb::infer::LayerNormParam> normLinearParam;
        normLinearParam.isAntiOutlier = FuzzUtil::GetRandomBool(fuzzIndex);
        normLinearParam.skipNorm = FuzzUtil::GetRandomBool(fuzzIndex);
        normLinearParam.normHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
        
        atb_speed::common::FusionLinearParam fusionLinearParam;
        fusionLinearParam.quantType = static_cast<atb_speed::common::LinearQuantType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_linearQuantTypeEnumTable, 6));
        fusionLinearParam.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);
        fusionLinearParam.hasBias = FuzzUtil::GetRandomBool(fuzzIndex);
        fusionLinearParam.transposeType = static_cast<atb_speed::common::TransposeType>(
            *(int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], -1, g_transposeTypeEnumTable, 3));
        normLinearParam.fusionLinearParam = fusionLinearParam;

        normLinearParam.normParamType.layerType = static_cast<atb::infer::LayerNormParam::LayerNormType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_normTypeEnumTable, 4));
        normLinearParam.normParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
        normLinearParam.normParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_quantTypeEnumTable, 6));

        normLinearParam.normQuantParamType.layerType = static_cast<atb::infer::LayerNormParam::LayerNormType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_normTypeEnumTable, 4));
        normLinearParam.normQuantParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
        normLinearParam.normQuantParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_quantTypeEnumTable, 6));

        atb::Node NormLinearNode;
        NormLinear(normLinearParam, &NormLinearNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(NormLinearRmsNormDTFuzz, NormLinear)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "NormLinearRmsNormDTFuzzNormLinearRmsNorm";

    DT_FUZZ_START(0, 10000, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        atb_speed::common::NormLinearParam<atb::infer::RmsNormParam> normLinearParam;
        normLinearParam.isAntiOutlier = FuzzUtil::GetRandomBool(fuzzIndex);
        normLinearParam.skipNorm = FuzzUtil::GetRandomBool(fuzzIndex);
        normLinearParam.normHasBias = FuzzUtil::GetRandomBool(fuzzIndex);

        atb_speed::common::FusionLinearParam fusionLinearParam;
        fusionLinearParam.quantType = static_cast<atb_speed::common::LinearQuantType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_linearQuantTypeEnumTable, 6));
        fusionLinearParam.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);
        fusionLinearParam.hasBias = FuzzUtil::GetRandomBool(fuzzIndex);
        fusionLinearParam.transposeType = static_cast<atb_speed::common::TransposeType>(
            *(int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], -1, g_transposeTypeEnumTable, 3));
        normLinearParam.fusionLinearParam = fusionLinearParam;

        normLinearParam.normParamType.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_normTypeEnumTable, 4));
        normLinearParam.normParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
        normLinearParam.normParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_quantTypeEnumTable, 6));

        normLinearParam.normQuantParamType.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_normTypeEnumTable, 4));
        normLinearParam.normQuantParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
        normLinearParam.normQuantParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
            *(unsigned int *)DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 0, g_quantTypeEnumTable, 6));

        atb::Node NormLinearNode;
        NormLinear(normLinearParam, &NormLinearNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}  // namespace atb_speed