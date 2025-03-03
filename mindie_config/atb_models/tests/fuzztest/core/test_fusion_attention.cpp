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
#include "operations/fusion/attention/fusion_attention.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
int g_packQuantTypeEnumFuzzTable[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

void attnGetFuzzParam(uint32_t &fuzzIndex, atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &param)
{
    // QKV linear param
    param.isGroupedQueryAttention = FuzzUtil::GetRandomBool(fuzzIndex);
    param.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);
    param.splitWithStride = FuzzUtil::GetRandomBool(fuzzIndex);
    param.qkvHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
    param.skipNorm = FuzzUtil::GetRandomBool(fuzzIndex);
    param.normHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
    param.enableNormQuantOp = FuzzUtil::GetRandomBool(fuzzIndex);

    param.packQuantType = static_cast<atb_speed::common::PackQuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, g_packQuantTypeEnumFuzzTable, 18)); // 18 is total number of pack quant type
    param.layerLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    param.layerLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };

    int RmsNormTypeEnumTable[] = {0, 1, 2, 3};
    param.normParamType.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, RmsNormTypeEnumTable, 4)); // 4 is total number of norm type
    param.normParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
    int QuantTypeEnumTable[] = {0, 1, 2, 3, 4, 5};
    param.normParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, QuantTypeEnumTable, 6)); // 6 is total number of quant type

    param.normQuantParamType.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, RmsNormTypeEnumTable, 4)); // 4 is total number of norm type
    param.normQuantParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
    param.normQuantParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, QuantTypeEnumTable, 6)); // 6 is total number of quant type
}

void attnGetSAFuzzParam(uint32_t &fuzzIndex, atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &param)
{
    // rope param
    int RotaryTypeEnumTable[] = {0, 1, 2};
    param.rotaryType = static_cast<atb_speed::common::RotaryType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, RotaryTypeEnumTable, 3)); // 3 is total number of rotary type

    // self attention param
    param.enableLogN = FuzzUtil::GetRandomBool(fuzzIndex);
    param.isFA = FuzzUtil::GetRandomBool(fuzzIndex);
    param.isPrefill = FuzzUtil::GetRandomBool(fuzzIndex);

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.batchRunStatusEnable = FuzzUtil::GetRandomBool(fuzzIndex);
    int CalcTypeEnumTable[] = {0, 1, 2, 3};
    selfAttentionParam.calcType = static_cast<atb::infer::SelfAttentionParam::CalcType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, CalcTypeEnumTable, 4)); // 4 is total number of calc type
    int KernelTypeEnumTable[] = {0, 1};
    selfAttentionParam.kernelType = static_cast<atb::infer::SelfAttentionParam::KernelType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, KernelTypeEnumTable, 2)); // 2 is total number of kernel type
    int ClampTypeEnumTable[] = {0, 1};
    selfAttentionParam.clampType = static_cast<atb::infer::SelfAttentionParam::ClampType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, ClampTypeEnumTable, 2)); // 2 is total number of clamp type
    int MaskTypeEnumTable[] = {0, 1, 2, 3, 4, 5, 6};
    selfAttentionParam.maskType = static_cast<atb::infer::SelfAttentionParam::MaskType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, MaskTypeEnumTable, 7)); // 7 is total number of mask type
    param.selfAttentionParam = selfAttentionParam;
}

void attnGetPAFuzzParam(uint32_t &fuzzIndex, atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &param)
{
    atb::infer::PagedAttentionParam pagedAttentionParam;
    int PAMaskTypeEnumTable[] = {0, 1, 2, 3};
    pagedAttentionParam.maskType = static_cast<atb::infer::PagedAttentionParam::MaskType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, PAMaskTypeEnumTable, 4)); // 4 is total number of mask type
    pagedAttentionParam.batchRunStatusEnable = FuzzUtil::GetRandomBool(fuzzIndex);
    int PAQuantTypeEnumTable[] = {0, 1};
    pagedAttentionParam.quantType = static_cast<atb::infer::PagedAttentionParam::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, PAQuantTypeEnumTable, 2)); // 2 is total number of quant type
    pagedAttentionParam.hasQuantOffset = FuzzUtil::GetRandomBool(fuzzIndex);
    int CompressTypeEnumTable[] = {0, 1};
    pagedAttentionParam.compressType = static_cast<atb::infer::PagedAttentionParam::CompressType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, CompressTypeEnumTable, 2)); // 2 is total number of compress type
    int PACalcTypeEnumTable[] = {0, 1};
    pagedAttentionParam.calcType = static_cast<atb::infer::PagedAttentionParam::CalcType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, PACalcTypeEnumTable, 2)); // 2 is total number of calc type
    param.pageAttentionParam = pagedAttentionParam;

    // self out linear param
    param.selfAttnHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
    param.supportLcoc = FuzzUtil::GetRandomBool(fuzzIndex);

    param.denseQuantType = static_cast<atb_speed::common::PackQuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, g_packQuantTypeEnumFuzzTable, 18)); // 18 is total number of pack quant type

    atb_speed::common::TensorParallelInfo tensorParallelInfo;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    tensorParallelInfo.rank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    tensorParallelInfo.worldSize = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    tensorParallelInfo.backend = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    param.selfOutLinearTensorParallelInfo = tensorParallelInfo;
}
TEST(FusionAttentionDTFuzz, Attention)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "FusionAttentionDTFuzzAttention";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> param;

        attnGetFuzzParam(fuzzIndex, param);
        attnGetSAFuzzParam(fuzzIndex, param);
        attnGetPAFuzzParam(fuzzIndex, param);
        
        atb::Node attentionNode;
        Attention(param, &attentionNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}