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
#include "models/llama/model/decoder_model.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

nlohmann::json GemmaGetParam(uint32_t &fuzzIndex)
{
    nlohmann::json parameter;
    parameter["skipWordEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableKvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["kvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["supportSpeculate"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["normEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 1024); // 128,1024: bound
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
    parameter["numHiddenLayers"] = numHiddenLayers;
    parameter["hiddenSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 128, 128, 8106); // 128,8106: bound
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    std::vector<std::vector<int>> modelPackQuantType;
    std::vector<std::vector<int>> modelLinearQuantType;
    std::vector<std::vector<int>> layerLinearTransposeType;
    for (int i = 0; i < numHiddenLayers; i++) { // 38 layNums
        modelPackQuantType.push_back({1, 1});
        modelLinearQuantType.push_back({0, -1, -1, 0, 0, -1, 0});
        layerLinearTransposeType.push_back({1, -1, -1, 1, 1, -1, 1});
    }
    parameter["packQuantType"] = modelPackQuantType;
    parameter["linearQuantType"] = modelLinearQuantType;
    parameter["linearTransposeType"] = layerLinearTransposeType;

    parameter["positionEmbeddingType"] = 0;
    parameter["normType"] = 0;
    parameter["rankTableFile"] = "";
    parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableAddNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
    return parameter;
}

nlohmann::json GemmaGetAclParam(uint32_t &fuzzIndex)
{
    nlohmann::json aclParameter;
    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 1,20: bound
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100 随机长度的最大值
    }
    aclParameter["seqLen"] = modelSeqLen;
    int tokenOffsetSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 1,20: bound
    std::vector<int> tokenOffsetLen;
    for (int i = 0; i < tokenOffsetSize; ++i) {
        tokenOffsetLen.push_back(std::rand() % 100); // 100 随机长度的最大值
    }
    aclParameter["tokenOffset"] = tokenOffsetLen;
    return aclParameter;
}

TEST(GemmaModelDTFuzz, GemmaModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "GemmaModelDTFuzzGemmaModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("telechat_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter = GemmaGetParam(fuzzIndex);
        aclParameter = GemmaGetAclParam(fuzzIndex);
    
        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object llamaFuzz = modelModule.attr("TelechatFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("gemma_GemmaDecoderModel");
            
            pybind11::object ret = llamaFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                llamaFuzzIns.attr("set_weight")();
                llamaFuzzIns.attr("set_kv_cache")();
                llamaFuzzIns.attr("execute")(acl_parameter_string, int(parameter["supportSpeculate"]));
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}