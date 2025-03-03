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
#include "operations/fusion/mlp/mlp.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void mlpGetRmsFuzzParam(uint32_t &fuzzIndex, atb_speed::common::MlpParam<atb::infer::RmsNormParam> &param)
{
    int LayerNormTypeEnumTable[] = {0, 1, 2, 3};
    param.normParamType.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, LayerNormTypeEnumTable, 4)); // 4 is total number of norm type
    param.normParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
    int QuantTypeEnumTable[] = {0, 1, 2, 3, 4, 5};
    param.normParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, QuantTypeEnumTable, 6)); // 6 is total number of quant type

    param.normQuantParamType.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, LayerNormTypeEnumTable, 4)); // 4 is total number of norm type
    param.normQuantParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
    param.normQuantParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, QuantTypeEnumTable, 6)); // 6 is total number of quant type

    atb::infer::ActivationParam activationParam;
    param.activationParam.activationType = static_cast<atb::infer::ActivationType>(
        *(unsigned int *) DT_SetGetNumberRange(
            &g_Element[fuzzIndex++], 0, 0, 8)); // 8 is total number of acttivation type
    
    atb_speed::common::TensorParallelInfo downLinearTensorParallelInfo;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    downLinearTensorParallelInfo.rank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    downLinearTensorParallelInfo.worldSize = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    downLinearTensorParallelInfo.backend = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    param.downLinearTensorParallelInfo = downLinearTensorParallelInfo;
}

void mlpGetLayerFuzzParam(uint32_t &fuzzIndex, atb_speed::common::MlpParam<atb::infer::LayerNormParam> &param)
{
    int LayerNormTypeEnumTable[] = {0, 1, 2, 3};
    param.normParamType.layerType = static_cast<atb::infer::LayerNormParam::LayerNormType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, LayerNormTypeEnumTable, 4)); // 4 is total number of norm type
    param.normParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
    int QuantTypeEnumTable[] = {0, 1, 2, 3, 4, 5};
    param.normParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, QuantTypeEnumTable, 6)); // 6 is total number of quant type

    param.normQuantParamType.layerType = static_cast<atb::infer::LayerNormParam::LayerNormType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, LayerNormTypeEnumTable, 4)); // 4 is total number of norm type
    param.normQuantParamType.normParam.epsilon = float(std::rand()) / RAND_MAX;
    param.normQuantParamType.normParam.quantType = static_cast<atb::infer::QuantType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, QuantTypeEnumTable, 6)); // 6 is total number of quant type

    atb::infer::ActivationParam activationParam;
    param.activationParam.activationType = static_cast<atb::infer::ActivationType>(
        *(unsigned int *) DT_SetGetNumberRange(
            &g_Element[fuzzIndex++], 0, 0, 8)); // 8 is total number of activation type
    
    atb_speed::common::TensorParallelInfo downLinearTensorParallelInfo;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    downLinearTensorParallelInfo.rank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    downLinearTensorParallelInfo.worldSize = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    downLinearTensorParallelInfo.backend = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    param.downLinearTensorParallelInfo = downLinearTensorParallelInfo;
}

template <typename NormParamType>
void mlpGetFuzzParam(uint32_t &fuzzIndex, atb_speed::common::MlpParam<NormParamType> &param)
{
    param.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);
    param.gateUpHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
    param.downHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
    param.supportLcoc = FuzzUtil::GetRandomBool(fuzzIndex);
    param.skipNorm = FuzzUtil::GetRandomBool(fuzzIndex);
    param.normHasBias = FuzzUtil::GetRandomBool(fuzzIndex);
    param.enableAddNorm = FuzzUtil::GetRandomBool(fuzzIndex);
    param.enableNormQuantOp = FuzzUtil::GetRandomBool(fuzzIndex);

    int MlpPackTypeEnumTable[] = {0, 1, 2};
    param.mlpPackType = static_cast<atb_speed::common::MlpPackType>(
        *(unsigned int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 0, MlpPackTypeEnumTable, 3)); // 3 is total number of pack type
    param.packQuantType = static_cast<atb_speed::common::PackQuantType>(
        *(unsigned int *) DT_SetGetNumberRange(
            &g_Element[fuzzIndex++], 0, 0, 17)); // 17 is total number of pack quant type
    param.downQuantType = static_cast<atb_speed::common::PackQuantType>(
        *(unsigned int *) DT_SetGetNumberRange(
            &g_Element[fuzzIndex++], 0, 0, 17)); // 17 is total number of pack quant type
    param.quantGroupSize = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 32, 32, 1024); // 32 is min, 1024 is max per group size

    std::vector<int> layerLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    param.layerLinearQuantType = layerLinearQuantType;
    std::vector<int> layerLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    param.layerLinearTransposeType = layerLinearTransposeType;
}

TEST(MlpDTFuzz, LayerNorm)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "MlpDTFuzzLayerNorm";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        atb_speed::common::MlpParam<atb::infer::LayerNormParam> param;
        
        mlpGetFuzzParam(fuzzIndex, param);
        mlpGetLayerFuzzParam(fuzzIndex, param);

        atb::Node MlpNode;
        Mlp(param, &MlpNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(MlpDTFuzz, RmsNorm)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "MlpDTFuzzRmsNorm";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        atb_speed::common::MlpParam<atb::infer::RmsNormParam> param;

        mlpGetFuzzParam(fuzzIndex, param);
        mlpGetRmsFuzzParam(fuzzIndex, param);

        atb::Node MlpNode;
        Mlp(param, &MlpNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}

template void mlpGetFuzzParam(uint32_t &fuzzIndex, atb_speed::common::MlpParam<atb::infer::LayerNormParam> &param);
template void mlpGetFuzzParam(uint32_t &fuzzIndex, atb_speed::common::MlpParam<atb::infer::RmsNormParam> &param);
}