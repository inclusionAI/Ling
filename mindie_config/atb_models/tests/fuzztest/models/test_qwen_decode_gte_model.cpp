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
#include "models/qwen/model/gte_decoder_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void GteQwenGetParamType(int numHiddenLayers, nlohmann::json &parameter)
{
    std::vector<int> layerPackQuantType = {
        std::rand() % 12,
        std::rand() % 12
    };
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;
    std::vector<int> LinearType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearQuantType.push_back(LinearType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearTransposeType.push_back(LinearType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
    LinearType.pop_back();
    std::vector<std::vector<int>> contain_six_vect;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        contain_six_vect.push_back(LinearType);
    }
    parameter["attnLinearQuantType"] = contain_six_vect;
    parameter["attnLinearTransposeType"] = contain_six_vect;
    LinearType.pop_back();
    LinearType.pop_back();
    std::vector<std::vector<int>> contain_four_vect;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        contain_four_vect.push_back(LinearType);
    }
    parameter["mlpLinearQuantType"] = contain_four_vect;
    parameter["gteLinearQuantType"] = contain_four_vect;
    parameter["mlpLinearTransposeType"] = contain_four_vect;
    parameter["gteLinearTransposeType"] = contain_four_vect;
}

nlohmann::json GteQwenGetParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json parameter;

    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["withEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["normEps"] = float(std::rand()) / RAND_MAX;
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    int hiddenSizeLowerBound = 128;
    int hiddenSizeUpperBound = 1024;
    parameter["hiddenSizePerAttentionHead"] =
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++],
                                          hiddenSizeLowerBound,
                                          hiddenSizeLowerBound,
                                          hiddenSizeUpperBound);
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank /
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["numHiddenLayers"] = numHiddenLayers;
    parameter["enableLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    int worldSize = numAttentionHeadsPerRank;
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    parameter["enableLogN"] = true;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    int numOfExperts = numAttentionHeadsPerRank;
    parameter["numOfExperts"] = numOfExperts;
    parameter["expertParallelDegree"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["maskStartIdx"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    parameter["numOfSelectedExperts"] = {};
    parameter["kvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["quantGroupSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enableSpeculate"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableSplitFuse"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLora"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["loraEnableGMM"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["normType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["isYarn"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["mscale"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableQScale"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLora"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isClassification"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["skipWordEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableAddNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableKvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["positionEmbeddingType"] = 0;
    GteQwenGetParamType(numHiddenLayers, parameter);
    return parameter;
}

nlohmann::json GteQwenGetAclParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json aclParameter;

    int layerTokenOffset = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelTokenOffset;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelTokenOffset.push_back(layerTokenOffset);
    }
    aclParameter["tokenOffset"] = modelTokenOffset;

    int layerSeqLen = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelSeqLen;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelSeqLen.push_back(layerSeqLen);
    }
    aclParameter["seqLen"] = modelSeqLen;

    return aclParameter;
}

TEST(GteQwenModelDTFuzz, DecoderModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "GteQwenModelDTFuzzDecoderModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("base_fuzz");
    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);

        nlohmann::json parameter, aclParameter;
        parameter = GteQwenGetParam(numHiddenLayers, fuzzIndex);
        aclParameter = GteQwenGetAclParam(numHiddenLayers, fuzzIndex);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object llamaFuzz = modelModule.attr("BaseFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("qwen_GteDecoderModel");

            pybind11::object ret = llamaFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                llamaFuzzIns.attr("set_weight")();
                llamaFuzzIns.attr("set_kv_cache")();
                llamaFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

}
