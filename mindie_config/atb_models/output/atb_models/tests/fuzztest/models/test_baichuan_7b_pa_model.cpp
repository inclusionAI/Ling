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
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void Baichuan7BSetParamType(int numHiddenLayers, nlohmann::json &parameter)
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

    std::vector<int> layerLinearQuantType = {
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
        modelLinearQuantType.push_back(layerLinearQuantType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<int> layerLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearTransposeType.push_back(layerLinearTransposeType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
    std::vector<std::vector<bool>> linearHasBias;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        linearHasBias.push_back({false, false, false, false});
    }
    parameter["linearHasBias"] = linearHasBias;
}

void Baichuan7BSetBaseParam(int numHiddenLayers, uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLite"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableSpeculate"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableCompressHead"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableSplitFuse"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLora"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["loraEnableGMM"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableKvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["kvQuantHasOffset"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableFA3"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableReduceQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableAddNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enablePrefixCache"] = FuzzUtil::GetRandomBool(fuzzIndex);
    int attnBackend = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    parameter["attnBackend"] = attnBackend;
    int positionEmbeddingType = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 2); // min: 0, max: 2
    parameter["positionEmbeddingType"] = positionEmbeddingType;
    parameter["normEps"] = float(std::rand()) / RAND_MAX;
    int normType = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 2);
    parameter["normType"] = normType;
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["quantGroupSize"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 64)); // min: 0, max: 64
    int hiddenSizeLowerBound = 128;
    int hiddenSizeUpperBound = 1024;
    parameter["hiddenSizePerAttentionHead"] =
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++],
                                          hiddenSizeLowerBound,
                                          hiddenSizeLowerBound,
                                          hiddenSizeUpperBound);
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank /
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["weightQuantType"] = "";
}

void Baichuan7BSetBaseModelParam(int numHiddenLayers, uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    parameter["skipWordEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["numHiddenLayers"] = numHiddenLayers;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["rankTableFile"] = "";
}

nlohmann::json Baichuan7BGetParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json parameter;

    Baichuan7BSetBaseParam(numHiddenLayers, fuzzIndex, parameter);
    Baichuan7BSetBaseModelParam(numHiddenLayers, fuzzIndex, parameter);
    Baichuan7BSetParamType(numHiddenLayers, parameter);

    return parameter;
}

nlohmann::json Baichuan7BGetAclParam(int numHiddenLayers, uint32_t &fuzzIndex)
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

TEST(Baichuan7BModelDTFuzz, PagedAttentionQuantModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "Baichuan7BModelDTFuzzPagedAttentionQuantModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("baichuan_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);

        nlohmann::json parameter, aclParameter;
        parameter = Baichuan7BGetParam(numHiddenLayers, fuzzIndex);
        aclParameter = Baichuan7BGetAclParam(numHiddenLayers, fuzzIndex);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object llamaFuzz = modelModule.attr("BaiChuanFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("baichuan2_7b_PagedAttentionQuantModel");

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
