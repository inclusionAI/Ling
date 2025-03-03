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
#include "models/internlm/20b/layer/internlm_decoder_layer.h"
#include "models/internlm/20b/model/internlm_decoder_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void Internlm20BGetParamType(int numHiddenLayers, nlohmann::json &parameter)
{
    bool isRandomTrue = std::rand() % 5 != 0;
    std::vector<int> layerPackQuantType = {
        std::rand() % 12,
        std::rand() % 12
    };
    std::vector<int> layerLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<int> layerLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    if (isRandomTrue) {
        layerPackQuantType = { 1, 1 };
        layerLinearQuantType = { 0, -1, -1, 0, 0, -1, 0 };
        layerLinearTransposeType = { 1, -1, -1, 1, 1, -1, 1 };
    }
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<std::vector<int>> modelLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearQuantType.push_back(layerLinearQuantType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearTransposeType.push_back(layerLinearTransposeType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
}

nlohmann::json Internlm20BGetParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json parameter;

    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    int normType = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 2);
    parameter["normEps"] = normType;
    parameter["normEps"] = float(std::rand()) / RAND_MAX;
    parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["skipWordEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["positionEmbeddingType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enablePrefixCache"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableKvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableFA3"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableSplitFuse"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableCompressHead"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLora"] = FuzzUtil::GetRandomBool(fuzzIndex);
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
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["enableSpeculate"] = false;

    Internlm20BGetParamType(numHiddenLayers, parameter);
    return parameter;
}

nlohmann::json Internlm20BGetAclParam(int numHiddenLayers, uint32_t &fuzzIndex)
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

TEST(Internlm20BDecodeModelDTFuzz, DecoderModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "Internlm20BDecodeModelDTFuzzDecoderModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("internlm_fuzz");

    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 80);

        nlohmann::json parameter, aclParameter;
        parameter = Internlm20BGetParam(numHiddenLayers, fuzzIndex);
        aclParameter = Internlm20BGetAclParam(numHiddenLayers, fuzzIndex);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object internlmFuzz = modelModule.attr("InternLMFuzz");
            pybind11::object internlmFuzzIns = internlmFuzz("internlm_20b_parallel_InternlmDecoderModel");

            pybind11::object ret = internlmFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                internlmFuzzIns.attr("set_weight")();
                internlmFuzzIns.attr("set_kv_cache")();
                internlmFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

}
