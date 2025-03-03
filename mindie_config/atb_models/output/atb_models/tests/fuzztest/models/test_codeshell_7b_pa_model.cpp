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
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

nlohmann::json CodeshellGetParam(uint32_t &fuzzIndex)
{
    nlohmann::json parameter;
    parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["normEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
    int normType = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    parameter["normType"] = normType;
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 1,20: bound
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 1024); // 128,1024: bound
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200); // 1,200: bound
    parameter["numHiddenLayers"] = numHiddenLayers;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 15); // 1,15: bound
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["takenOffset"] = {1};
    parameter["seqLen"] = {1};
    
    std::vector<std::vector<int>> modelPackQuantType;
    FuzzUtil::GetRandomModelType(modelPackQuantType, 2, numHiddenLayers, 12); // 2: len, 12: quant
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<std::vector<int>> modelLinearQuantType;
    FuzzUtil::GetRandomModelType(modelLinearQuantType, 7, numHiddenLayers, 3); // 7: len, 3: quant
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<std::vector<int>> modelLinearTransposeType;
    FuzzUtil::GetRandomModelType(modelLinearTransposeType, 7, numHiddenLayers, 3); // 7: len, 3: quant
    parameter["linearTransposeType"] = modelLinearTransposeType;

    return parameter;
}

nlohmann::json CodeshellGetAclParam(uint32_t &fuzzIndex)
{
    nlohmann::json aclParameter;
    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 1,20: bound
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100: mod
    }
    aclParameter["seqLen"] = modelSeqLen;

    return aclParameter;
}

TEST(CodeshellModelDTFuzz, CodeshellModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("CodeshellModelDTFuzz Begin");
    std::string fuzzName = "CodeshellModelDTFuzzCodeshellModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("codeshell_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        nlohmann::json parameter, aclParameter;
        parameter = CodeshellGetParam(fuzzIndex);
        aclParameter = CodeshellGetAclParam(fuzzIndex);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object codeshellFuzz = modelModule.attr("CodeshellFuzz");
            pybind11::object codeshellFuzzIns = codeshellFuzz("codeshell_7b_DecoderModel");
            
            pybind11::object ret = codeshellFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                codeshellFuzzIns.attr("set_weight")();
                codeshellFuzzIns.attr("set_kv_cache")();
                codeshellFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}